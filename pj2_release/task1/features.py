"""词形 (word shape) 特征函数。

OOV 处理思路：训练时把低频词替换为 ``shape(word)``（例如 ``Bradford``
→ ``<INITCAP>``），把诸如大小写、数字、连字符、后缀等"形态信号"作为
HMM 的伪 token 喂入。这样推理时未登录词依然能落到一个有 emission 概率的
具体 tag，避免被 ``<UNK>`` 均匀化。

参考：Brants (1999) TnT Tagger / Bikel et al. NER 中常用的 OOV 处理方式。
"""

from __future__ import annotations

import re
from typing import Callable, Iterable

# ----------------------------- English ---------------------------------

_RE_NUM = re.compile(r"^\d+$")
_RE_DECIMAL = re.compile(r"^\d+[\.,]\d+$")
_RE_ORDINAL = re.compile(r"^\d+(?:st|nd|rd|th)$", re.IGNORECASE)
_RE_NUMERIC = re.compile(r"^[\d\-/.,:]+$")
_RE_INITIALS = re.compile(r"^(?:[A-Z]\.){2,}$")
_RE_INITCAP = re.compile(r"^[A-Z][a-z]+$")
_RE_ALLCAP = re.compile(r"^[A-Z]+$")
_RE_MIXEDCAP = re.compile(r"^[A-Z][a-zA-Z]*[A-Z][a-zA-Z]*$")

# 后缀越靠前优先级越高
_ENGLISH_SUFFIXES = (
    "tion", "ness", "ment", "able", "ous",
    "ing", "ed", "ly", "er",
)


def english_word_shape(w: str) -> str:
    """把单词映射到一个紧凑的"形态类别"字符串。"""
    if not w:
        return "<EMPTY>"

    # 数字 / 数值
    if _RE_NUM.match(w):
        return "<NUM>"
    if _RE_DECIMAL.match(w):
        return "<DECIMAL>"
    if _RE_ORDINAL.match(w):
        return "<ORDINAL>"
    if _RE_NUMERIC.match(w):
        return "<NUMERIC>"

    # 大小写 / 缩写
    if _RE_INITIALS.match(w):
        return "<INITIALS>"
    if _RE_ALLCAP.match(w):
        return "<ALLCAP>" if len(w) > 1 else "<SINGLECAP>"
    if _RE_INITCAP.match(w):
        return "<INITCAP>"
    if _RE_MIXEDCAP.match(w):
        return "<MIXEDCAP>"

    # 连字符复合词
    if "-" in w:
        return "<HYPHEN-CAP>" if w[:1].isupper() else "<HYPHEN>"

    # 词缀
    wl = w.lower()
    for suf in _ENGLISH_SUFFIXES:
        if wl.endswith(suf) and len(wl) > len(suf):
            return f"<SUF-{suf.upper()}>"

    # 首字母大写但不属于上面的规则（例如 "Mr"、"De"）
    if w[:1].isupper():
        return "<CAP>"

    # 全小写普通词
    return "<UNK>"


# ----------------------------- Chinese ---------------------------------

_HANZI_RE = re.compile(r"^[\u4e00-\u9fff\u3400-\u4dbf]$")


def chinese_word_shape(w: str) -> str:
    """中文是字符级 token，按 unicode block 区分汉字 / 字母 / 数字 / 标点。"""
    if not w:
        return "<EMPTY>"
    if len(w) == 1:
        c = w
        if _HANZI_RE.match(c):
            return "<HANZI>"
        if c.isdigit():
            return "<DIG>"
        if c.isascii() and c.isalpha():
            return "<ENU>" if c.isupper() else "<EN>"
        return "<PUNC>"
    # 多字符（少见，但比如全角空格、特殊符号组合）
    if all(_HANZI_RE.match(ch) for ch in w):
        return "<HANZI-MULTI>"
    return "<UNK>"


# ----------------------------- 共用 ------------------------------------


def detect_language(token_sample: Iterable[str]) -> str:
    """启发式：含汉字视为 Chinese，否则 English。"""
    for w in token_sample:
        if any(_HANZI_RE.match(c) for c in w):
            return "Chinese"
    return "English"


def get_shape_fn(language: str) -> Callable[[str], str]:
    if language == "Chinese":
        return chinese_word_shape
    if language == "English":
        return english_word_shape
    raise ValueError(f"未知语言: {language}")
