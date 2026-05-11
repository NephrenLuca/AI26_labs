"""数据读写工具：CoNLL 格式数据集加载、预测结果输出。

CoNLL 风格的 NER 数据：每行 ``token<space>tag``，句子之间用空行分隔。
本模块与 task1/data_utils.py 对齐，确保两阶段数据接口一致。
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple


def read_conll(path: str) -> List[Tuple[List[str], List[str]]]:
    """读取 CoNLL 格式文件，返回 ``[(tokens, tags), ...]``。

    - 跳过空行作为句子分隔符
    - 自动跳过完全空白的多余分隔行
    """
    sentences: List[Tuple[List[str], List[str]]] = []
    cur_tokens: List[str] = []
    cur_tags: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n").rstrip("\r")
            if line.strip() == "":
                if cur_tokens:
                    sentences.append((cur_tokens, cur_tags))
                    cur_tokens, cur_tags = [], []
                continue
            parts = line.split(" ")
            if len(parts) < 2:
                token, tag = parts[0], "O"
            else:
                # 容错：token 内含空格时取最后一段为 tag
                token = " ".join(parts[:-1])
                tag = parts[-1]
            cur_tokens.append(token)
            cur_tags.append(tag)
    if cur_tokens:
        sentences.append((cur_tokens, cur_tags))
    return sentences


# 行的占位类型：None 表示空行，否则保存原始 (token, gold_tag)
LineRecord = Optional[Tuple[str, str]]


def read_lines_with_blanks(path: str) -> List[LineRecord]:
    """逐行读取，保留空行（用 None 表示），便于按原格式输出预测。"""
    records: List[LineRecord] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n").rstrip("\r")
            if line.strip() == "":
                records.append(None)
                continue
            parts = line.split(" ")
            if len(parts) < 2:
                token, tag = parts[0], "O"
            else:
                token = " ".join(parts[:-1])
                tag = parts[-1]
            records.append((token, tag))
    return records


def group_into_sentences(records: Sequence[LineRecord]) -> List[List[str]]:
    """根据空行分组成 token 序列列表，丢弃空行结构。"""
    sentences: List[List[str]] = []
    cur: List[str] = []
    for rec in records:
        if rec is None:
            if cur:
                sentences.append(cur)
                cur = []
        else:
            cur.append(rec[0])
    if cur:
        sentences.append(cur)
    return sentences


def write_predictions(
    out_path: str,
    records: Sequence[LineRecord],
    predictions: Sequence[Sequence[str]],
) -> None:
    """按照原始文件行结构写出预测：保留空行位置，token 不变，标签替换为预测值。

    要求 ``predictions`` 中所有标签数与 ``records`` 中非空行数相等。
    """
    flat_pred_iter = iter(tag for sent in predictions for tag in sent)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for rec in records:
            if rec is None:
                f.write("\n")
            else:
                token, _gold = rec
                try:
                    pred = next(flat_pred_iter)
                except StopIteration as exc:  # pragma: no cover - defensive
                    raise ValueError(
                        "预测标签数量少于原文件中的非空行数。"
                    ) from exc
                f.write(f"{token} {pred}\n")
    leftover = list(flat_pred_iter)
    if leftover:
        raise ValueError(
            f"预测标签数量({len(leftover)} 多余)与原文件非空行数不一致。"
        )


def detect_language(sentences: Sequence[Tuple[List[str], List[str]]]) -> str:
    """启发式：含汉字视为 Chinese，否则 English。"""
    import re
    hanzi = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")
    for tokens, _ in sentences[:50]:
        for t in tokens:
            if hanzi.search(t):
                return "Chinese"
    return "English"
