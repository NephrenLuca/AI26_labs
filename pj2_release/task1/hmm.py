"""手写 HMM (一阶 / 二阶) 用于命名实体识别。

设计要点
---------
- ``order=1`` : 一阶 bigram HMM
- ``order=2`` : 二阶 trigram HMM，Brants 1999 deleted-interpolation 自动学 λ
- **TnT 后缀树 OOV (Brants 2000)**：低频词在递归平滑的后缀树上估计
  ``p(t | suffix_k(w))``，按贝叶斯翻成相对发射 ``log p(t|s) - log p(t)``，
  以 ``weight_tnt`` 为系数加性叠加在固定 shape 列的 log MLE 上
- **shape vocab**：训练时把训练集里所有词的"主"shape (最长合规后缀
  ``<SUF$branch$k$suf>`` 或固定类别 ``<CAT$...>``) 也作为 vocab 词条；
  推理时 OOV 走 shape 列
- BIO/BMES 结构化转移约束（自动识别方案）
- π / A / B **三套独立 Laplace α**
- numpy / cupy 双后端
"""

from __future__ import annotations

import pickle
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from constraints import build_masks
from features import detect_language
from suffix_model import SuffixShapeModel, TnTSuffixTree, is_capitalized


# ----------------------------- backend ---------------------------------


def resolve_backend(device: str) -> Tuple[Any, str]:
    """``cpu`` -> numpy；``cuda`` -> cupy。CuPy/CUDA 不可用时回退 CPU。"""
    device = device.lower()
    if device == "cpu":
        return np, "cpu"
    if device in ("cuda", "gpu"):
        try:
            import os as _os

            import cupy as cp  # type: ignore[import-not-found]

            n_dev = cp.cuda.runtime.getDeviceCount()
            if n_dev <= 0:
                raise RuntimeError("未检测到 CUDA 设备。")
            cp.cuda.Device(0).use()
            visible = _os.environ.get("CUDA_VISIBLE_DEVICES", "<all>")
            try:
                props = cp.cuda.runtime.getDeviceProperties(0)
                name = props["name"].decode() if isinstance(
                    props["name"], (bytes, bytearray)
                ) else str(props["name"])
            except Exception:  # noqa: BLE001
                name = "?"
            print(
                f"[HMM] 使用 CuPy / CUDA："
                f"CUDA_VISIBLE_DEVICES={visible}，"
                f"可见 {n_dev} 张 GPU，激活 device 0 ({name})。"
            )
            return cp, "cuda"
        except Exception as exc:  # noqa: BLE001
            print(
                f"[HMM] CuPy / CUDA 不可用 ({type(exc).__name__}: {exc})，"
                "已自动回退到 CPU。"
            )
            return np, "cpu"
    raise ValueError(f"未知的 device: {device!r}，请使用 'cpu' 或 'cuda'。")


def _to_numpy(arr: Any) -> np.ndarray:
    if isinstance(arr, np.ndarray):
        return arr
    if hasattr(arr, "get"):
        return arr.get()
    return np.asarray(arr)


# ----------------------------- HMM -------------------------------------


class HMM:
    """监督式 HMM。"""

    def __init__(
        self,
        device: str = "cpu",
        order: int = 2,
        alpha_pi: float = 1.0,
        alpha_trans: float = 0.01,
        alpha_emit: float = 1e-4,
        min_word_freq: int = 2,
        # OOV / 后缀 shape
        use_suffix_oov: bool = True,
        rare_threshold: int = 10,
        max_suffix_len: int = 10,
        suffix_min_count: int = 5,
        # A1：TnT 后缀树
        use_tnt_tree: bool = True,
        theta_tnt: float = 1.0,
        weight_tnt: float = 0.3,
        # 结构约束
        use_constraints: bool = True,
        language: str = "auto",
    ) -> None:
        self.xp, self.device = resolve_backend(device)
        if order not in (1, 2):
            raise ValueError("order 必须是 1 或 2。")
        self.order = int(order)
        self.alpha_pi = float(alpha_pi)
        self.alpha_trans = float(alpha_trans)
        self.alpha_emit = float(alpha_emit)
        self.min_word_freq = int(min_word_freq)
        self.use_suffix_oov = bool(use_suffix_oov)
        self.rare_threshold = int(rare_threshold)
        self.max_suffix_len = int(max_suffix_len)
        self.suffix_min_count = int(suffix_min_count)
        self.use_tnt_tree = bool(use_tnt_tree)
        self.theta_tnt = float(theta_tnt)
        self.weight_tnt = float(weight_tnt)
        self.use_constraints = bool(use_constraints)
        self.language = language

        self.tag2id: Dict[str, int] = {}
        self.id2tag: List[str] = []
        self.word2id: Dict[str, int] = {}
        self.id2word: List[str] = []
        self.word_freq: np.ndarray = np.zeros(0, dtype=np.float64)
        # 标记 vocab 里每条是否是 shape 条目（vs 真实 token）
        self.is_shape: np.ndarray = np.zeros(0, dtype=bool)
        self.scheme: Optional[str] = None
        self.lambdas: Optional[Tuple[float, float, float]] = None
        self.suffix_model: Optional[SuffixShapeModel] = None
        self.tnt_tree: Optional[TnTSuffixTree] = None

        self.log_pi: Any = None
        self.log_trans: Any = None
        self.log_emit: Any = None
        self.log_pi2: Any = None
        self.log_trans2: Any = None
        self._log_emit_np: Optional[np.ndarray] = None  # CPU 副本（用于 OOV 查找）

    # ------------------------------------------------------------------
    # vocab
    # ------------------------------------------------------------------
    def _build_vocab(
        self, sentences: Sequence[Tuple[Sequence[str], Sequence[str]]]
    ) -> Counter:
        tag_set: set[str] = set()
        word_count: Counter[str] = Counter()
        for tokens, tags in sentences:
            tag_set.update(tags)
            word_count.update(tokens)

        self.id2tag = sorted(tag_set)
        self.tag2id = {t: i for i, t in enumerate(self.id2tag)}

        if self.language == "auto":
            self.language = detect_language(list(word_count.keys())[:200])

        # ---- 训练 SuffixShapeModel（按首字母大小写分支提供"主"shape token）----
        if self.use_suffix_oov:
            rare_words: List[str] = [
                w for w, c in word_count.items() if c <= self.rare_threshold
            ]
            split_cap = (self.language == "English")
            self.suffix_model = SuffixShapeModel(
                max_suffix_len=self.max_suffix_len,
                min_count=self.suffix_min_count,
                split_capitalization=split_cap,
                language=self.language,
            )
            self.suffix_model.fit(rare_words)

        # ---- 词表：真实词 (freq ≥ min_word_freq) ∪ 所有 shape 词条 ----
        kept_words: List[str] = [
            w for w, c in word_count.items() if c >= self.min_word_freq
        ]
        shape_tokens: List[str] = (
            self.suffix_model.all_shape_tokens(list(word_count.keys()))
            if self.suffix_model
            else []
        )
        self.id2word = kept_words + shape_tokens
        self.word2id = {w: i for i, w in enumerate(self.id2word)}
        self.is_shape = np.array(
            [False] * len(kept_words) + [True] * len(shape_tokens), dtype=bool
        )
        self.word_freq = np.array(
            [word_count.get(w, 0) for w in self.id2word], dtype=np.float64
        )
        if self.suffix_model is not None:
            print(
                f"[HMM] Vocab: real_words={len(kept_words)}, "
                f"shape_tokens={len(shape_tokens)}, "
                f"max_suffix_len={self.max_suffix_len}, "
                f"min_count={self.suffix_min_count}"
            )
        return word_count

    # ------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------
    def fit(
        self, sentences: Sequence[Tuple[Sequence[str], Sequence[str]]]
    ) -> None:
        word_count = self._build_vocab(sentences)
        T = len(self.id2tag)
        V = len(self.id2word)

        pi_count = np.zeros(T, dtype=np.float64)
        trans_count = np.zeros((T, T), dtype=np.float64)
        # **单张** emission 计数表：每个 token 同时贡献到 word_id 列与 shape_id
        # 列。这样所有列共享同一个分母（按 tag 行归一化），P(w|t) 与 P(shape|t)
        # 在同一概率尺度上 —— blending 时不会出现尺度灾难。
        emit_count = np.zeros((T, V), dtype=np.float64)
        pi2_count = (
            np.zeros((T, T), dtype=np.float64) if self.order == 2 else None
        )
        trans2_count = (
            np.zeros((T, T, T), dtype=np.float64) if self.order == 2 else None
        )

        # 收集低频词的 (word, tag_id) 对，用于训练 TnT 后缀树（A1）
        rare_pairs: List[Tuple[str, int]] = []

        for tokens, tags in sentences:
            if not tokens:
                continue
            tag_ids = [self.tag2id[t] for t in tags]

            pi_count[tag_ids[0]] += 1
            for prev_t, cur_t in zip(tag_ids[:-1], tag_ids[1:]):
                trans_count[prev_t, cur_t] += 1

            for w, tid in zip(tokens, tag_ids):
                w_id = self.word2id.get(w)
                if w_id is not None:
                    emit_count[tid, w_id] += 1
                elif self.suffix_model is not None:
                    suf_tok = self.suffix_model.shape_of(w)
                    cat_tok = self.suffix_model.category_shape_of(w)
                    seen_ids: set = set()
                    for tok in (suf_tok, cat_tok):
                        s_id = self.word2id.get(tok)
                        if s_id is not None and s_id not in seen_ids:
                            emit_count[tid, s_id] += 1
                            seen_ids.add(s_id)
                # 低频词进 TnT 后缀树训练集
                if (
                    self.use_tnt_tree
                    and word_count[w] <= self.rare_threshold
                ):
                    rare_pairs.append((w, tid))

            if self.order == 2:
                if len(tag_ids) >= 2:
                    pi2_count[tag_ids[0], tag_ids[1]] += 1
                for i in range(2, len(tag_ids)):
                    trans2_count[tag_ids[i - 2], tag_ids[i - 1], tag_ids[i]] += 1

        # ---- 训练 TnT 后缀树（A1）----
        if self.use_tnt_tree and rare_pairs:
            split_cap = (self.language == "English")
            self.tnt_tree = TnTSuffixTree(
                num_tags=T,
                max_suffix_len=max(self.max_suffix_len, 1),
                theta=self.theta_tnt,
                split_capitalization=split_cap,
            )
            self.tnt_tree.fit(rare_pairs)
            print(
                f"[HMM] TnT 后缀树: rare_pairs={len(rare_pairs)}, "
                f"max_len={self.tnt_tree.max_suffix_len}, "
                f"θ_tnt={self.theta_tnt}, weight_tnt={self.weight_tnt}"
            )

        ap, at, ae = self.alpha_pi, self.alpha_trans, self.alpha_emit
        pi = (pi_count + ap) / (pi_count.sum() + ap * T)
        trans = (trans_count + at) / (
            trans_count.sum(axis=1, keepdims=True) + at * T
        )
        log_pi = np.log(pi).astype(np.float32)
        log_trans = np.log(trans).astype(np.float32)

        # 单张 P 矩阵：所有列共享同一行归一化。
        # 已见词读 log P(w | t)；OOV 在推理时读对应 shape 列的 log P(shape | t)。
        emit_p = (emit_count + ae) / (
            emit_count.sum(axis=1, keepdims=True) + ae * V
        )
        log_emit = np.log(np.maximum(emit_p, 1e-30)).astype(np.float32)

        # ---- 二阶平滑 ----
        log_pi2 = log_trans2 = None
        if self.order == 2:
            assert pi2_count is not None and trans2_count is not None
            uni_count = pi_count + trans_count.sum(axis=0)
            uni_count[uni_count < 1] = 1.0  # 防 0
            total_tokens = float(uni_count.sum())
            eps = 1e-8
            uni_p = (uni_count + eps) / (total_tokens + eps * T)
            bi_p = (trans_count + eps) / (
                trans_count.sum(axis=1, keepdims=True) + eps * T
            )
            tri_p = (trans2_count + eps) / (
                trans2_count.sum(axis=2, keepdims=True) + eps * T
            )
            l1, l2, l3 = self._deleted_interpolation(
                trans2_count, trans_count, uni_count, total_tokens
            )
            self.lambdas = (l1, l2, l3)
            interp = (
                l3 * tri_p + l2 * bi_p[None, :, :] + l1 * uni_p[None, None, :]
            )
            log_trans2 = np.log(np.maximum(interp, 1e-30)).astype(np.float32)
            pi2_mle = (pi2_count + eps) / (
                pi2_count.sum(axis=1, keepdims=True) + eps * T
            )
            lam = max(0.5, l3)
            pi2_p = lam * pi2_mle + (1.0 - lam) * bi_p
            log_pi2 = np.log(np.maximum(pi2_p, 1e-30)).astype(np.float32)

        # ---- 结构化约束 ----
        if self.use_constraints:
            start_mask, trans_mask, self.scheme = build_masks(self.id2tag)
            log_pi = log_pi + start_mask
            log_trans = log_trans + trans_mask
            if self.order == 2:
                log_pi2 = log_pi2 + start_mask[:, None] + trans_mask
                log_trans2 = log_trans2 + trans_mask[None, :, :]
        else:
            self.scheme = None

        self.log_pi = self.xp.asarray(log_pi)
        self.log_trans = self.xp.asarray(log_trans)
        self.log_emit = self.xp.asarray(log_emit)
        self._log_emit_np = log_emit
        if self.order == 2:
            self.log_pi2 = self.xp.asarray(log_pi2)
            self.log_trans2 = self.xp.asarray(log_trans2)

    @staticmethod
    def _deleted_interpolation(
        c3: np.ndarray, c2: np.ndarray, c1: np.ndarray, total: float
    ) -> Tuple[float, float, float]:
        l1 = l2 = l3 = 0.0
        nz_a, nz_b, nz_c = np.nonzero(c3)
        for a, b, c in zip(nz_a.tolist(), nz_b.tolist(), nz_c.tolist()):
            cabc = c3[a, b, c]
            cab = c2[a, b]
            cb = c1[b]
            cc = c1[c]
            cbc = c2[b, c]
            f3 = (cabc - 1) / (cab - 1) if cab > 1 else 0.0
            f2 = (cbc - 1) / (cb - 1) if cb > 1 else 0.0
            f1 = (cc - 1) / (total - 1) if total > 1 else 0.0
            if f3 >= f2 and f3 >= f1:
                l3 += cabc
            elif f2 >= f1:
                l2 += cabc
            else:
                l1 += cabc
        s = l1 + l2 + l3
        if s == 0:
            return 0.05, 0.25, 0.7
        return l1 / s, l2 / s, l3 / s

    # ------------------------------------------------------------------
    # per-position emission（含 OOV shape 查找 + A1 TnT 加性项）
    # ------------------------------------------------------------------
    def _emission_for_token(self, w: str) -> np.ndarray:
        """返回 [T] 的 log emission 向量。"""
        T = len(self.id2tag)
        if w in self.word2id:
            return self._log_emit_np[:, self.word2id[w]]
        if self.suffix_model is not None:
            s_tok = self.suffix_model.shape_of(w)
            s_id = self.word2id.get(s_tok)
            if s_id is not None:
                base = self._log_emit_np[:, s_id]
                tnt = self._tnt_logp(w)
                return base + self.weight_tnt * tnt if tnt is not None else base
        return np.zeros(T, dtype=np.float32)

    def _tnt_logp(self, w: str) -> Optional[np.ndarray]:
        """返回 ``log p(t|suffix(w)) - log p(t)`` 向量；不可用时返回 None。"""
        if self.tnt_tree is None or self.weight_tnt == 0:
            return None
        return self.tnt_tree.emit_logprob_oov(w)

    def _compute_batch_emission(
        self, batch_tokens: Sequence[Sequence[str]], L_max: int
    ) -> Any:
        T = len(self.id2tag)
        B = len(batch_tokens)
        emit = np.zeros((B, L_max, T), dtype=np.float32)
        cache: Dict[str, np.ndarray] = {}
        for b, sent in enumerate(batch_tokens):
            for l, w in enumerate(sent):
                vec = cache.get(w)
                if vec is None:
                    vec = self._emission_for_token(w)
                    cache[w] = vec
                emit[b, l] = vec
        return self.xp.asarray(emit)

    # ------------------------------------------------------------------
    # Viterbi
    # ------------------------------------------------------------------
    def _viterbi_batch_order1(
        self, batch_tokens: Sequence[Sequence[str]]
    ) -> List[List[str]]:
        xp = self.xp
        T = int(self.log_pi.shape[0])
        B = len(batch_tokens)
        lengths = [len(s) for s in batch_tokens]
        L_max = max(lengths)

        emit_full = self._compute_batch_emission(batch_tokens, L_max)
        log_pi_b = xp.broadcast_to(self.log_pi[None, :], (B, T))
        dp = log_pi_b + emit_full[:, 0, :]
        backpointer = xp.zeros((B, L_max, T), dtype=xp.int32)
        len_arr = xp.asarray(np.asarray(lengths, dtype=np.int64))

        for i in range(1, L_max):
            scores = dp[:, :, None] + self.log_trans[None, :, :]
            best_prev = xp.argmax(scores, axis=1).astype(xp.int32)
            best_score = xp.max(scores, axis=1)
            new_dp = best_score + emit_full[:, i, :]
            mask = (i < len_arr)[:, None]
            dp = xp.where(mask, new_dp, dp)
            backpointer[:, i, :] = best_prev

        return self._backtrack_order1(dp, backpointer, lengths)

    def _backtrack_order1(self, dp, backpointer, lengths):
        bp_np = _to_numpy(backpointer)
        dp_np = _to_numpy(dp)
        results: List[List[str]] = []
        for b, L in enumerate(lengths):
            if L == 0:
                results.append([])
                continue
            last = int(np.argmax(dp_np[b]))
            path = [last]
            for i in range(L - 1, 0, -1):
                last = int(bp_np[b, i, last])
                path.append(last)
            path.reverse()
            results.append([self.id2tag[s] for s in path])
        return results

    def _viterbi_batch_order2(
        self, batch_tokens: Sequence[Sequence[str]]
    ) -> List[List[str]]:
        xp = self.xp
        T = int(self.log_pi.shape[0])
        B = len(batch_tokens)
        lengths = [len(s) for s in batch_tokens]
        L_max = max(lengths)

        emit_full = self._compute_batch_emission(batch_tokens, L_max)
        log_pi = self.log_pi
        log_pi2 = self.log_pi2
        log_trans2 = self.log_trans2

        if L_max >= 2:
            dp = (
                log_pi[None, :, None]
                + emit_full[:, 0, :, None]
                + log_pi2[None, :, :]
                + emit_full[:, 1, None, :]
            )
        else:
            dp = log_pi[None, :] + emit_full[:, 0, :]

        backpointer = xp.zeros((B, L_max, T, T), dtype=xp.int32)
        len_arr = xp.asarray(np.asarray(lengths, dtype=np.int64))

        for i in range(2, L_max):
            scores = dp[:, :, :, None] + log_trans2[None, :, :, :]
            best_pp = xp.argmax(scores, axis=1).astype(xp.int32)
            best_score = xp.max(scores, axis=1)
            new_dp = best_score + emit_full[:, i, None, :]
            mask = (i < len_arr)[:, None, None]
            dp = xp.where(mask, new_dp, dp) if dp.ndim == 3 else new_dp
            backpointer[:, i, :, :] = best_pp

        return self._backtrack_order2(dp, backpointer, emit_full, lengths)

    def _backtrack_order2(self, dp, backpointer, emit_full, lengths):
        bp_np = _to_numpy(backpointer)
        log_pi_np = _to_numpy(self.log_pi)
        emit_np = _to_numpy(emit_full)
        T = int(self.log_pi.shape[0])

        dp_np = _to_numpy(dp)
        results: List[List[str]] = []
        for b, L in enumerate(lengths):
            if L == 0:
                results.append([])
                continue
            if L == 1:
                scores1 = log_pi_np + emit_np[b, 0]
                results.append([self.id2tag[int(np.argmax(scores1))]])
                continue
            mat = dp_np[b]
            if mat.ndim == 1:
                results.append([self.id2tag[int(np.argmax(mat))]])
                continue
            flat = int(mat.argmax())
            y_prev, y_cur = divmod(flat, T)
            path: List[int] = [0] * L
            path[L - 1] = y_cur
            path[L - 2] = y_prev
            for i in range(L - 1, 1, -1):
                pp = int(bp_np[b, i, y_prev, y_cur])
                path[i - 2] = pp
                y_cur, y_prev = y_prev, pp
            results.append([self.id2tag[s] for s in path])
        return results

    # ------------------------------------------------------------------
    def viterbi_batch(self, batch_tokens):
        if not batch_tokens:
            return []
        if max((len(s) for s in batch_tokens), default=0) == 0:
            return [[] for _ in batch_tokens]
        if self.order == 1:
            return self._viterbi_batch_order1(batch_tokens)
        return self._viterbi_batch_order2(batch_tokens)

    def viterbi(self, tokens):
        return self.viterbi_batch([tokens])[0]

    def predict(self, sentences, batch_size: int = 64):
        if not sentences:
            return []
        order = sorted(range(len(sentences)), key=lambda i: len(sentences[i]))
        results: List[List[str]] = [[] for _ in sentences]
        for start in range(0, len(order), batch_size):
            chunk = order[start : start + batch_size]
            batch = [sentences[i] for i in chunk]
            preds = self.viterbi_batch(batch)
            for i, pred in zip(chunk, preds):
                results[i] = pred
        return results

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        assert self.log_pi is not None, "未训练的模型不能保存。"
        blob = {
            "tag2id": self.tag2id,
            "id2tag": self.id2tag,
            "word2id": self.word2id,
            "id2word": self.id2word,
            "is_shape": self.is_shape,
            "word_freq": self.word_freq,
            "log_pi": _to_numpy(self.log_pi),
            "log_trans": _to_numpy(self.log_trans),
            "log_emit": _to_numpy(self.log_emit),
            "order": self.order,
            "alpha_pi": self.alpha_pi,
            "alpha_trans": self.alpha_trans,
            "alpha_emit": self.alpha_emit,
            "min_word_freq": self.min_word_freq,
            "use_suffix_oov": self.use_suffix_oov,
            "rare_threshold": self.rare_threshold,
            "max_suffix_len": self.max_suffix_len,
            "suffix_min_count": self.suffix_min_count,
            "use_tnt_tree": self.use_tnt_tree,
            "theta_tnt": self.theta_tnt,
            "weight_tnt": self.weight_tnt,
            "tnt_tree": self.tnt_tree,
            "use_constraints": self.use_constraints,
            "language": self.language,
            "scheme": self.scheme,
            "lambdas": self.lambdas,
            "suffix_model": self.suffix_model,
        }
        if self.order == 2:
            blob["log_pi2"] = _to_numpy(self.log_pi2)
            blob["log_trans2"] = _to_numpy(self.log_trans2)
        with open(path, "wb") as f:
            pickle.dump(blob, f)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "HMM":
        with open(path, "rb") as f:
            blob = pickle.load(f)
        m = cls(
            device=device,
            order=blob.get("order", 1),
            alpha_pi=blob.get("alpha_pi", 1.0),
            alpha_trans=blob.get("alpha_trans", 0.01),
            alpha_emit=blob.get("alpha_emit", 1e-4),
            min_word_freq=blob.get("min_word_freq", 2),
            use_suffix_oov=blob.get("use_suffix_oov", True),
            rare_threshold=blob.get("rare_threshold", 10),
            max_suffix_len=blob.get("max_suffix_len", 10),
            suffix_min_count=blob.get("suffix_min_count", 5),
            use_tnt_tree=blob.get("use_tnt_tree", True),
            theta_tnt=blob.get("theta_tnt", 1.0),
            weight_tnt=blob.get("weight_tnt", 0.3),
            use_constraints=blob.get("use_constraints", True),
            language=blob.get("language", "auto"),
        )
        m.tag2id = blob["tag2id"]
        m.id2tag = blob["id2tag"]
        m.word2id = blob["word2id"]
        m.id2word = blob["id2word"]
        m.is_shape = blob.get("is_shape", np.zeros(len(m.id2word), dtype=bool))
        m.word_freq = blob.get("word_freq", np.zeros(len(m.id2word), dtype=np.float64))
        m.scheme = blob.get("scheme")
        m.lambdas = blob.get("lambdas")
        m.suffix_model = blob.get("suffix_model")
        m.tnt_tree = blob.get("tnt_tree")
        m._log_emit_np = blob["log_emit"]
        m.log_pi = m.xp.asarray(blob["log_pi"])
        m.log_trans = m.xp.asarray(blob["log_trans"])
        m.log_emit = m.xp.asarray(blob["log_emit"])
        if m.order == 2:
            m.log_pi2 = m.xp.asarray(blob["log_pi2"])
            m.log_trans2 = m.xp.asarray(blob["log_trans2"])
        return m
