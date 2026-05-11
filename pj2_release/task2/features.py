"""CRF 特征工程：CRF++-style 模板 + 形态学（shape）特征。

设计要点
--------
- **稀疏二值特征**：每个 token 位置上提取若干字符串特征 ``"<TPL>=<value>"``，
  通过 :class:`FeatureVocab` 映射成整数 id。
- **CRF++ 模板**：参考 ``NER/template_for_crf.utf8``（Unigram 部分），覆盖
  ``token[t-2..t+2]`` 单元与若干二元组合。
- **形态学特征**（shape）：英文字母大小写 / 是否数字 / 后缀；中文字符类别（汉字/数字/字母/标点）。
- **min_count 阈值**：低频特征会被丢弃，相当于 L0 正则；显著缩减词表，缓解过拟合。
- **OOV**：测试时未登录的特征字符串会查无此 id（返回 -1），collate 时被过滤；
  位置上若所有特征都被过滤完，会回填 ``<EMPTY>``（id=0）以保证 EmbeddingBag 正常工作。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple

# 句首/句尾占位 token，使得边界附近窗口特征仍能正常触发
BOS_2 = "<BOS-2>"
BOS_1 = "<BOS-1>"
EOS_1 = "<EOS+1>"
EOS_2 = "<EOS+2>"


# ============================ 形态学（shape） ===============================

_RE_NUM = re.compile(r"^\d+$")
_RE_DECIMAL = re.compile(r"^\d+[\.,]\d+$")
_RE_ORDINAL = re.compile(r"^\d+(?:st|nd|rd|th)$", re.IGNORECASE)
_RE_NUMERIC = re.compile(r"^[\d\-/.,:]+$")
_RE_INITIALS = re.compile(r"^(?:[A-Z]\.){2,}$")
_RE_INITCAP = re.compile(r"^[A-Z][a-z]+$")
_RE_ALLCAP = re.compile(r"^[A-Z]+$")
_RE_MIXEDCAP = re.compile(r"^[A-Z][a-zA-Z]*[A-Z][a-zA-Z]*$")
_HANZI_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")

_ENGLISH_SUFFIXES = (
    "tion", "ness", "ment", "able", "ous",
    "ing", "ed", "ly", "er", "ist",
)


def english_shape(w: str) -> str:
    """英文 token 的紧凑形态类别。"""
    if not w:
        return "EMPTY"
    if _RE_NUM.match(w):
        return "NUM"
    if _RE_DECIMAL.match(w):
        return "DECIMAL"
    if _RE_ORDINAL.match(w):
        return "ORDINAL"
    if _RE_NUMERIC.match(w):
        return "NUMERIC"
    if _RE_INITIALS.match(w):
        return "INITIALS"
    if _RE_ALLCAP.match(w):
        return "ALLCAP" if len(w) > 1 else "SINGLECAP"
    if _RE_INITCAP.match(w):
        return "INITCAP"
    if _RE_MIXEDCAP.match(w):
        return "MIXEDCAP"
    if "-" in w:
        return "HYPHEN-CAP" if w[:1].isupper() else "HYPHEN"
    wl = w.lower()
    for suf in _ENGLISH_SUFFIXES:
        if wl.endswith(suf) and len(wl) > len(suf):
            return f"SUF-{suf.upper()}"
    if w[:1].isupper():
        return "CAP"
    return "LOWER"


def chinese_shape(w: str) -> str:
    """中文（字符级）token 的形态类别。"""
    if not w:
        return "EMPTY"
    if len(w) == 1:
        c = w
        if _HANZI_RE.match(c):
            return "HANZI"
        if c.isdigit():
            return "DIG"
        if c.isascii() and c.isalpha():
            return "ENU" if c.isupper() else "EN"
        return "PUNC"
    if all(_HANZI_RE.match(ch) for ch in w):
        return "HANZI-MULTI"
    return "OTHER"


# ============================ 特征词表 ====================================


class FeatureVocab:
    """管理特征字符串 → 整数 id 的映射。

    - id 0 永远保留给 ``<EMPTY>``（用作 EmbeddingBag 的占位 token，必要时回填）。
    - 训练阶段调用 :meth:`build` 一次性扫描；推理阶段通过 :meth:`get` 查询，
      未登录返回 -1（调用方自行过滤）。
    """

    EMPTY = "<EMPTY>"

    def __init__(self) -> None:
        self.feat2id: Dict[str, int] = {self.EMPTY: 0}
        self.frozen: bool = False

    def __len__(self) -> int:
        return len(self.feat2id)

    def get(self, feat: str) -> int:
        return self.feat2id.get(feat, -1)

    def add(self, feat: str) -> int:
        if feat in self.feat2id:
            return self.feat2id[feat]
        if self.frozen:
            return -1
        idx = len(self.feat2id)
        self.feat2id[feat] = idx
        return idx

    def build(self, all_feats: Iterable[Iterable[str]], min_count: int = 2) -> None:
        """从训练集统计 + 阈值过滤，构造最终 feat2id。

        ``all_feats``: 形如 ``[[feat1, feat2, ...], [feat3, ...], ...]``
        每个内部列表对应一个位置上提取出来的特征字符串。
        """
        from collections import Counter

        cnt: Counter = Counter()
        for feats in all_feats:
            cnt.update(feats)
        # 重置：仅保留 EMPTY，再加入阈值通过的特征
        self.feat2id = {self.EMPTY: 0}
        for feat, c in cnt.items():
            if c >= min_count:
                self.feat2id[feat] = len(self.feat2id)
        self.frozen = True


# ============================ 模板特征提取 =================================


@dataclass
class FeatureExtractor:
    """逐 token 提取 CRF++-style 特征。

    Attributes
    ----------
    language : str
        ``"Chinese"`` 或 ``"English"``。
    use_shape : bool
        是否加入形态学特征。
    use_affix : bool
        是否加入前/后缀特征（英文长度 2-3，中文按字符级 token 通常关闭）。
    max_affix_len : int
        前/后缀最长长度（仅英文有效）。
    """

    language: str
    use_shape: bool = True
    use_affix: bool = True
    max_affix_len: int = 3
    vocab: FeatureVocab = field(default_factory=FeatureVocab)

    # ---------------- 内部工具 ----------------

    def _shape(self, w: str) -> str:
        if self.language == "Chinese":
            return chinese_shape(w)
        return english_shape(w)

    @staticmethod
    def _ctx(tokens: Sequence[str], i: int, off: int) -> str:
        """返回 ``tokens[i+off]``，越界用 BOS/EOS 占位。"""
        j = i + off
        if j < 0:
            return BOS_2 if j == -2 else BOS_1
        if j >= len(tokens):
            return EOS_2 if j - len(tokens) == 1 else EOS_1
        return tokens[j]

    # ---------------- 主接口 ----------------

    def extract_position(self, tokens: Sequence[str], i: int) -> List[str]:
        """提取位置 ``i`` 处的特征字符串列表（与 CRF++ Unigram 模板对齐）。"""
        feats: List[str] = []
        # 单 token 特征：t-2, t-1, t, t+1, t+2
        for off, name in (
            (-2, "U00"), (-1, "U01"), (0, "U02"), (1, "U03"), (2, "U04"),
        ):
            feats.append(f"{name}={self._ctx(tokens, i, off)}")
        # 二元组合：(-2,-1), (-1,0), (-1,+1), (0,+1), (+1,+2)
        for (a, b), name in (
            ((-2, -1), "U05"), ((-1, 0), "U06"), ((-1, 1), "U07"),
            ((0, 1), "U08"), ((1, 2), "U09"),
        ):
            feats.append(
                f"{name}={self._ctx(tokens, i, a)}/{self._ctx(tokens, i, b)}"
            )
        # 形态学特征（位置 t-1, t, t+1 的 shape）
        if self.use_shape:
            for off, name in ((-1, "S01"), (0, "S02"), (1, "S03")):
                feats.append(f"{name}={self._shape(self._ctx(tokens, i, off))}")
            # 当前 token 的小写形式（仅英文有意义；中文小写就是它自己）
            if self.language == "English":
                feats.append(f"LOW={self._ctx(tokens, i, 0).lower()}")
        # 前/后缀特征（仅英文长度>1时）
        if self.use_affix and self.language == "English":
            cur = self._ctx(tokens, i, 0)
            if not cur.startswith("<"):
                for L in range(2, self.max_affix_len + 1):
                    if len(cur) > L:
                        feats.append(f"PRE{L}={cur[:L].lower()}")
                        feats.append(f"SUF{L}={cur[-L:].lower()}")
        return feats

    def extract_sentence(self, tokens: Sequence[str]) -> List[List[str]]:
        return [self.extract_position(tokens, i) for i in range(len(tokens))]

    # ---------------- 训练词表构建 ----------------

    def build_vocab(
        self,
        sentences: Sequence[Tuple[List[str], List[str]]],
        min_count: int = 2,
    ) -> None:
        """扫描训练集所有特征，按阈值过滤后构建 ``self.vocab``。"""
        all_feats: List[List[str]] = []
        for tokens, _ in sentences:
            for i in range(len(tokens)):
                all_feats.append(self.extract_position(tokens, i))
        self.vocab.build(all_feats, min_count=min_count)

    def feat_ids(self, tokens: Sequence[str]) -> List[List[int]]:
        """提取并映射成整数 id 列表（OOV 已被过滤），用于直接喂给模型。"""
        out: List[List[int]] = []
        for i in range(len(tokens)):
            feats = self.extract_position(tokens, i)
            ids = [fid for fid in (self.vocab.get(f) for f in feats) if fid >= 0]
            if not ids:
                ids = [0]  # 回填 <EMPTY>，防止 EmbeddingBag 出现空 bag
            out.append(ids)
        return out


# ============================ 标签词表 ====================================


def build_tag_vocab(
    sentences: Sequence[Tuple[List[str], List[str]]],
    canonical: Sequence[str] | None = None,
) -> Tuple[List[str], Dict[str, int]]:
    """构建标签词表。

    若提供 ``canonical``（如 check.py 的 sorted_labels），则按其顺序排列；
    训练数据中出现但不在 canonical 内的标签会追加在后面（理论上不应发生）。
    """
    seen = set()
    for _, tags in sentences:
        seen.update(tags)
    if canonical is not None:
        ordered = [t for t in canonical if t in seen or t == "O"]
        for t in seen:
            if t not in ordered:
                ordered.append(t)
    else:
        ordered = sorted(seen)
    return ordered, {t: i for i, t in enumerate(ordered)}
