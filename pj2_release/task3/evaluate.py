"""与 ``NER/check.py`` 等价的评测函数（UTF-8 兼容）。

直接复用 ``task1/evaluate.py`` 的实现细节：
* 标签集与 sklearn micro/macro 计算方式与 ``check.py`` 严格一致；
* 仅在打开文件时显式指定 UTF-8 编码，避免 Windows 默认 GBK 解码失败。
"""

from __future__ import annotations

import warnings
from typing import List, Sequence

from sklearn import metrics

warnings.filterwarnings("ignore")


SORTED_LABELS_ENG: List[str] = [
    "O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC",
]

SORTED_LABELS_CHN: List[str] = [
    "O",
    "B-NAME", "M-NAME", "E-NAME", "S-NAME",
    "B-CONT", "M-CONT", "E-CONT", "S-CONT",
    "B-EDU", "M-EDU", "E-EDU", "S-EDU",
    "B-TITLE", "M-TITLE", "E-TITLE", "S-TITLE",
    "B-ORG", "M-ORG", "E-ORG", "S-ORG",
    "B-RACE", "M-RACE", "E-RACE", "S-RACE",
    "B-PRO", "M-PRO", "E-PRO", "S-PRO",
    "B-LOC", "M-LOC", "E-LOC", "S-LOC",
]


def _load_pairs(gold_path: str, my_path: str) -> tuple:
    y_true: List[str] = []
    y_pred: List[str] = []
    with open(gold_path, "r", encoding="utf-8") as g_f, open(
        my_path, "r", encoding="utf-8"
    ) as m_f:
        g_lines = g_f.readlines()
        m_lines = m_f.readlines()
        n = min(len(g_lines), len(m_lines))
        for i in range(n):
            if g_lines[i].strip() == "":
                continue
            g_parts = g_lines[i].rstrip("\n").rstrip("\r").split(" ")
            m_parts = m_lines[i].rstrip("\n").rstrip("\r").split(" ")
            y_true.append(g_parts[-1])
            y_pred.append(m_parts[-1])
    return y_true, y_pred


def evaluate(language: str, gold_path: str, my_path: str) -> str:
    """打印并返回 sklearn classification_report 字符串（与 check.py 一致）。"""
    sort_labels = SORTED_LABELS_ENG if language == "English" else SORTED_LABELS_CHN
    y_true, y_pred = _load_pairs(gold_path, my_path)
    report = metrics.classification_report(
        y_true=y_true, y_pred=y_pred, labels=sort_labels[1:], digits=4
    )
    print(report)
    return report


def micro_f1(language: str, gold_path: str, my_path: str) -> float:
    """直接返回 micro avg F1（用于训练中选择 best 模型）。"""
    sort_labels = SORTED_LABELS_ENG if language == "English" else SORTED_LABELS_CHN
    y_true, y_pred = _load_pairs(gold_path, my_path)
    return float(
        metrics.f1_score(
            y_true=y_true,
            y_pred=y_pred,
            labels=sort_labels[1:],
            average="micro",
            zero_division=0,
        )
    )
