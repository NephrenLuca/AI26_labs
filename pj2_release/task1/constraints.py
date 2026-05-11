"""BIO / BMES 标签结构化约束。

HMM 学到的转移矩阵是软概率，会出现 ``O→I-PER`` / ``B-PER→I-ORG`` /
``B-ORG→M-EDU`` 等结构上不可能的转移。把它们的 log 概率加上一个很大的负数
(NEG ≈ -1e9) 就能在 Viterbi 中被排除。

支持两种标签方案，自动检测：
- BIO  : ``O / B-X / I-X``                 (英文数据)
- BMES : ``O / B-X / M-X / E-X / S-X``     (中文数据)
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

NEG = -1e9  # 等价于 -inf 但避免后续算术出现 nan


def detect_scheme(id2tag: List[str]) -> str:
    has_M = any(t.startswith("M-") for t in id2tag)
    has_E = any(t.startswith("E-") for t in id2tag)
    if has_M and has_E:
        return "BMES"
    return "BIO"


def _parse(tag: str) -> Tuple[str, str | None]:
    if tag == "O":
        return ("O", None)
    if "-" in tag:
        prefix, etype = tag.split("-", 1)
        return (prefix, etype)
    return (tag, None)


def build_masks(id2tag: List[str]) -> Tuple[np.ndarray, np.ndarray, str]:
    """返回 ``(start_mask, trans_mask, scheme)``，两个 mask 都是要被加到 log 概率上的。

    - ``start_mask[t] = NEG`` 表示 tag t 不能作为句首
    - ``trans_mask[i, j] = NEG`` 表示 ``i -> j`` 在结构上不允许
    """
    T = len(id2tag)
    scheme = detect_scheme(id2tag)
    start_mask = np.zeros(T, dtype=np.float32)
    trans_mask = np.zeros((T, T), dtype=np.float32)

    parsed = [_parse(t) for t in id2tag]

    if scheme == "BIO":
        # I-X 不能作为句首
        for i, (pi, _ei) in enumerate(parsed):
            if pi == "I":
                start_mask[i] = NEG
        # I-X 只能跟在 B-X 或 I-X (同实体类型) 后面
        for i, (pi, ei) in enumerate(parsed):
            for j, (pj, ej) in enumerate(parsed):
                if pj == "I":
                    if pi not in ("B", "I") or ei != ej:
                        trans_mask[i, j] = NEG
        return start_mask, trans_mask, scheme

    # BMES
    # M-X / E-X 不能作为句首
    for i, (pi, _ei) in enumerate(parsed):
        if pi in ("M", "E"):
            start_mask[i] = NEG

    # 双向规则：
    # (a) 如果当前是 M/E，前一个必须是同类型的 B/M
    # (b) 如果前一个是 B/M，当前必须是同类型的 M/E
    for i, (pi, ei) in enumerate(parsed):
        for j, (pj, ej) in enumerate(parsed):
            if pj in ("M", "E"):
                if pi not in ("B", "M") or ei != ej:
                    trans_mask[i, j] = NEG
            if pi in ("B", "M"):
                if pj not in ("M", "E") or ej != ei:
                    trans_mask[i, j] = NEG

    return start_mask, trans_mask, scheme
