"""TnT-style suffix shape + 固定类别兜底 + Brants 后缀树平滑。

本模块提供两种 OOV 表示形式：

A) **shape token** —— 每个词被映射到一个字符串（``<SUF$...>`` 或
   ``<CAT$...>``）。可作为伪 vocab 条目参与 HMM 普通发射的 MLE，
   尺度与已见词一致。
B) **TnT 后缀树** —— Brants (2000) 风格的多层平滑：
   ``p(t|s_k) = (count(t,s_k) + θ * p(t|s_{k-1})) / (count(s_k) + θ)``，
   推理时按 Bayes 翻成相对发射 ``log p(t|s) - log p(t)``。

英文按首字母大小写分两支（``upper`` / ``lower``）。中文不分支。
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np

from features import chinese_word_shape, english_word_shape


def is_capitalized(w: str) -> bool:
    return bool(w) and w[0].isupper()


class SuffixShapeModel:
    """组合：TnT 最长后缀 + 固定类别兜底。"""

    def __init__(
        self,
        max_suffix_len: int = 10,
        min_count: int = 5,
        split_capitalization: bool = True,
        language: str = "English",
    ) -> None:
        self.max_suffix_len = int(max_suffix_len)
        self.min_count = int(min_count)
        self.split_cap = bool(split_capitalization)
        self.language = language
        # valid[branch] = list of sets per length
        self.valid: Dict[str, List[Set[str]]] = {}

    def _branch_key(self, w: str) -> str:
        if not self.split_cap:
            return "any"
        return "upper" if is_capitalized(w) else "lower"

    def _category_shape(self, w: str) -> str:
        """落到固定类别的兜底名（与 features.py 等价但加分支前缀避免撞名）。"""
        if self.language == "Chinese":
            return f"<CAT${chinese_word_shape(w)}>"
        return f"<CAT${english_word_shape(w)}>"

    # ------------------------------------------------------------------
    def fit(self, rare_words: Sequence[str]) -> None:
        """收集低频词的后缀，确定哪些后缀有足够支持。"""
        K = self.max_suffix_len
        if self.split_cap:
            buckets: Dict[str, List[str]] = {"upper": [], "lower": []}
            for w in rare_words:
                buckets[self._branch_key(w)].append(w)
        else:
            buckets = {"any": list(rare_words)}

        for branch, words in buckets.items():
            cnt: List[Counter] = [Counter() for _ in range(K + 1)]
            for w in words:
                if not w:
                    continue
                for k in range(1, min(K, len(w)) + 1):
                    cnt[k][w[-k:]] += 1
            valid_per_k: List[Set[str]] = [set() for _ in range(K + 1)]
            for k in range(1, K + 1):
                for suf, c in cnt[k].items():
                    if c >= self.min_count:
                        valid_per_k[k].add(suf)
            self.valid[branch] = valid_per_k

        if self.split_cap:
            for needed in ("upper", "lower"):
                self.valid.setdefault(needed, [set() for _ in range(K + 1)])
        else:
            self.valid.setdefault("any", [set() for _ in range(K + 1)])

    # ------------------------------------------------------------------
    def shape_of(self, w: str) -> str:
        """返回 token 的"主"shape：优先 TnT 后缀，否则固定类别兜底。"""
        if not w:
            return self._category_shape(w)
        branch = self._branch_key(w)
        valid = self.valid.get(branch)
        if valid is not None:
            K = min(self.max_suffix_len, len(w))
            for k in range(K, 0, -1):
                suf = w[-k:]
                if suf in valid[k]:
                    return f"<SUF${branch}${k}${suf}>"
        return self._category_shape(w)

    def category_shape_of(self, w: str) -> str:
        """单独暴露固定类别 shape，可同时贡献到 vocab 中。"""
        return self._category_shape(w)

    # ------------------------------------------------------------------
    def all_shape_tokens(self, all_tokens: Sequence[str]) -> List[str]:
        """枚举所有 shape token —— 包括"主"shape 与固定类别兜底，便于 vocab。"""
        out: Set[str] = set()
        for w in all_tokens:
            out.add(self.shape_of(w))
            out.add(self.category_shape_of(w))
        return sorted(out)


# ----------------------------------------------------------------------
# Brants 2000 — 真正的递归平滑后缀树
# ----------------------------------------------------------------------


class TnTSuffixTree:
    """Brants 2000 后缀树 OOV 发射估计器。

    训练完后 ``emit_logprob_oov(w)`` 返回长度为 ``num_tags`` 的相对发射向量
    ``log p(t | suffix(w)) - log p(t)``（per-position 上多出的 ``log p(suffix(w))``
    常数与 t 无关，对 Viterbi argmax 无影响）。
    """

    def __init__(
        self,
        num_tags: int,
        max_suffix_len: int = 10,
        theta: float = 1.0,
        split_capitalization: bool = True,
    ) -> None:
        self.num_tags = num_tags
        self.max_suffix_len = int(max_suffix_len)
        self.theta = float(theta)
        self.split_cap = bool(split_capitalization)

        # smooth[branch][k][suffix] -> np.ndarray[T] (smoothed P(t | s_k))
        self.smooth: Dict[str, List[Dict[str, np.ndarray]]] = {}
        self.tag_prior: Dict[str, np.ndarray] = {}

    def _branch_key(self, w: str) -> str:
        if not self.split_cap:
            return "any"
        return "upper" if is_capitalized(w) else "lower"

    def fit(self, rare_pairs: Sequence[Tuple[str, int]]) -> None:
        T = self.num_tags
        K = self.max_suffix_len

        if self.split_cap:
            buckets: Dict[str, List[Tuple[str, int]]] = {"upper": [], "lower": []}
            for w, t in rare_pairs:
                buckets[self._branch_key(w)].append((w, t))
        else:
            buckets = {"any": list(rare_pairs)}

        for branch, pairs in buckets.items():
            self._fit_branch(branch, pairs, T, K)

        if self.split_cap:
            for needed in ("upper", "lower"):
                if needed not in self.smooth:
                    self.smooth[needed] = [dict() for _ in range(K + 1)]
                    self.smooth[needed][0][""] = np.ones(T) / T
                    self.tag_prior[needed] = np.ones(T) / T
        elif "any" not in self.smooth:
            self.smooth["any"] = [dict() for _ in range(K + 1)]
            self.smooth["any"][0][""] = np.ones(T) / T
            self.tag_prior["any"] = np.ones(T) / T

    def _fit_branch(
        self, branch: str, pairs: Sequence[Tuple[str, int]], T: int, K: int
    ) -> None:
        tag_total = np.zeros(T, dtype=np.float64)
        cnt: List[Dict[str, np.ndarray]] = [
            defaultdict(lambda: np.zeros(T, dtype=np.float64))
            for _ in range(K + 1)
        ]
        for w, tid in pairs:
            tag_total[tid] += 1
            cnt[0][""][tid] += 1
            for k in range(1, min(K, len(w)) + 1):
                cnt[k][w[-k:]][tid] += 1

        total = float(tag_total.sum())
        prior = tag_total / total if total > 0 else np.ones(T) / T

        smooth: List[Dict[str, np.ndarray]] = [dict() for _ in range(K + 1)]
        smooth[0][""] = prior.copy()
        for k in range(1, K + 1):
            for suf, c in cnt[k].items():
                parent_suf = suf[1:] if k > 1 else ""
                parent = smooth[k - 1].get(parent_suf, prior)
                tot = float(c.sum())
                smooth[k][suf] = (c + self.theta * parent) / (tot + self.theta)

        self.smooth[branch] = smooth
        self.tag_prior[branch] = prior

    # ------------------------------------------------------------------
    def emit_logprob_oov(self, w: str) -> np.ndarray:
        """返回 ``log p(t | suffix(w)) - log p(t)``，长度为 ``num_tags``。"""
        if not w:
            return np.zeros(self.num_tags, dtype=np.float32)
        branch = self._branch_key(w)
        smooth = self.smooth.get(branch)
        prior = self.tag_prior.get(branch)
        if smooth is None or prior is None:
            return np.zeros(self.num_tags, dtype=np.float32)

        K = min(self.max_suffix_len, len(w))
        for k in range(K, 0, -1):
            suf = w[-k:]
            if suf in smooth[k]:
                p = smooth[k][suf]
                return (
                    np.log(np.clip(p, 1e-12, None))
                    - np.log(np.clip(prior, 1e-12, None))
                ).astype(np.float32)
        return np.zeros(self.num_tags, dtype=np.float32)
