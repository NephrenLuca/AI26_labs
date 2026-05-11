"""跨平台兼容的评测函数。

与 ``NER/check.py`` 的输出严格一致 (使用相同的 ``sorted_labels`` 与
``sklearn.metrics.classification_report``)，区别仅在于显式以 UTF-8 打开文件，
避免 Windows 默认 GBK 解码失败。Linux 服务器上结果完全相同。
"""

from __future__ import annotations

import warnings
from typing import List

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


def evaluate(language: str, gold_path: str, my_path: str) -> str:
    """对应 check.py.check —— 打印并返回 sklearn classification_report 字符串。"""
    sort_labels = SORTED_LABELS_ENG if language == "English" else SORTED_LABELS_CHN
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
            g_tag = g_parts[-1]
            m_tag = m_parts[-1]
            y_true.append(g_tag)
            y_pred.append(m_tag)
    report = metrics.classification_report(
        y_true=y_true, y_pred=y_pred, labels=sort_labels[1:], digits=4
    )
    print(report)
    return report
