"""数据读写 + BERT 子词对齐 + Dataset / Collator。

设计要点
--------
1. **CoNLL 数据 IO**：与 ``task1/data_utils.py`` 完全对齐 —— ``read_conll``、
   ``read_lines_with_blanks``、``group_into_sentences``、``write_predictions``。
2. **句子切片** (``chunk_sentences``)：BERT 单次最长 512 子词，过长句子被切
   成若干 ``Chunk``；每个 chunk 记录所属句子 id 与起始位置，便于推理后按句重
   组预测结果。
3. **子词对齐** (``encode_chunk``)：用 ``tokenizer(words, is_split_into_words=
   True)`` 分词；用 ``word_ids()`` 找到每个原 word 的“第一个子词”位置，模型
   仅在这些位置取隐藏向量喂给 CRF。
4. **NERDataset + collate_fn**：返回适合 CRF 的 batch（``input_ids``,
   ``attention_mask``, ``first_subword_idx``, ``word_mask``, ``labels`` 等），
   全部为 ``torch.LongTensor``。

Pad 策略
--------
* 子词级别：``input_ids`` 与 ``attention_mask`` 按 batch 内最长 *子词数* pad；
* 单词级别：``first_subword_idx``、``word_mask``、``labels`` 按 batch 内最长
  *词数* pad。pad 的 ``first_subword_idx`` 全设为 0（任意有效索引即可，因为
  ``word_mask`` 会将其屏蔽）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset


# ----------------------------------------------------------------------
# CoNLL IO（与 task1 对齐）
# ----------------------------------------------------------------------


def read_conll(path: str) -> List[Tuple[List[str], List[str]]]:
    """读取 CoNLL 格式文件，返回 ``[(tokens, tags), ...]``。"""
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
                token = " ".join(parts[:-1])
                tag = parts[-1]
            cur_tokens.append(token)
            cur_tags.append(tag)
    if cur_tokens:
        sentences.append((cur_tokens, cur_tags))
    return sentences


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
    """按原始文件行结构写出预测：保留空行位置，token 不变，标签替换为预测值。"""
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
                except StopIteration as exc:  # pragma: no cover
                    raise ValueError("预测标签数量少于原文件中的非空行数。") from exc
                f.write(f"{token} {pred}\n")
    leftover = list(flat_pred_iter)
    if leftover:
        raise ValueError(
            f"预测标签数量({len(leftover)} 多余)与原文件非空行数不一致。"
        )


# ----------------------------------------------------------------------
# 标签集
# ----------------------------------------------------------------------


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


def get_label_list(language: str) -> List[str]:
    """返回该语言的标签集（含 O）。"""
    if language == "English":
        return list(SORTED_LABELS_ENG)
    if language == "Chinese":
        return list(SORTED_LABELS_CHN)
    raise ValueError(f"未知 language: {language}")


def build_label_maps(labels: Sequence[str]) -> Tuple[Dict[str, int], List[str]]:
    label2id = {lab: i for i, lab in enumerate(labels)}
    id2label = list(labels)
    return label2id, id2label


# ----------------------------------------------------------------------
# 句子分块
# ----------------------------------------------------------------------


@dataclass
class Chunk:
    """一段连续的词 + 标签，源自某个句子的 [word_start : word_start+len(words)]。"""

    sentence_idx: int
    word_start: int
    words: List[str]
    labels: Optional[List[int]] = None  # 训练时为 int label id，推理时为 None


def _subword_lens_per_word(tokenizer, words: Sequence[str]) -> List[int]:
    """逐 word 用 tokenizer 计算 *不含 special token* 的子词数；至少为 1。

    某些 token 可能被 tokenizer 编码为 0 个子词（control char / lower 后空），
    在这种情况下我们仍计为 1（后续会塞 [UNK]），便于精确预算。
    """
    lens: List[int] = []
    for w in words:
        try:
            ids = tokenizer.encode(w, add_special_tokens=False)
        except Exception:
            ids = [0]
        n = max(1, len(ids))
        lens.append(n)
    return lens


def chunk_sentences(
    sentences: Sequence[Sequence[str]],
    max_words_per_chunk: int,
    sentence_labels: Optional[Sequence[Sequence[int]]] = None,
    tokenizer=None,
    max_seq_length: int = 512,
) -> List[Chunk]:
    """按最大词数 + 子词预算切片句子。

    * 若 ``tokenizer`` 为 None，仅按 ``max_words_per_chunk`` 切片（粗粒度，
      要求调用方保证不会触发子词截断）。
    * 若提供 ``tokenizer``，则同时根据 ``max_seq_length`` 预算每个 chunk 的
      子词数（含 ``[CLS]/[SEP]`` 共 2 个 special token）：贪心打包，直到加入
      下一个 word 会让总子词数超 ``max_seq_length`` 为止，再起一个新 chunk。
      这样保证任意合法切片都不会发生截断丢词。
    """
    if max_words_per_chunk <= 0:
        raise ValueError("max_words_per_chunk 必须 > 0")
    chunks: List[Chunk] = []
    sub_budget = max(1, max_seq_length - 2)  # 留给 CLS / SEP

    for si, words in enumerate(sentences):
        labels = sentence_labels[si] if sentence_labels is not None else None
        if len(words) == 0:
            continue
        if tokenizer is None:
            sub_lens = None
        else:
            sub_lens = _subword_lens_per_word(tokenizer, words)

        i = 0
        N = len(words)
        while i < N:
            # 贪心确定本 chunk 的结束位置
            j = i
            cur_sub = 0
            while j < N and (j - i) < max_words_per_chunk:
                add = sub_lens[j] if sub_lens is not None else 1
                if sub_lens is not None and add > sub_budget:
                    # 极端情况：单个 word 子词数 > budget；硬塞，让 encode_chunk
                    # 走截断路径（极少；相当于该 word 只用前几个子词的发射）
                    if j == i:
                        j += 1
                    break
                if sub_lens is not None and cur_sub + add > sub_budget:
                    break
                cur_sub += add
                j += 1
            if j == i:
                # 防御：永远至少前进一个 word
                j = i + 1
            seg_words = list(words[i:j])
            seg_labels = list(labels[i:j]) if labels is not None else None
            chunks.append(
                Chunk(
                    sentence_idx=si,
                    word_start=i,
                    words=seg_words,
                    labels=seg_labels,
                )
            )
            i = j
    return chunks


# ----------------------------------------------------------------------
# 子词级编码
# ----------------------------------------------------------------------


@dataclass
class EncodedChunk:
    """分词后的 chunk：对应 ``Chunk`` 的子词级张量字段（仍然是 list）。"""

    sentence_idx: int
    word_start: int
    n_words: int
    input_ids: List[int]
    attention_mask: List[int]
    first_subword_idx: List[int]  # 长度 = n_words；指向 input_ids 中该词第一个子词的位置
    labels: Optional[List[int]]
    fallback_used: List[bool] = field(default_factory=list)  # 仅诊断用


def encode_chunk(
    chunk: Chunk,
    tokenizer,
    max_seq_length: int,
) -> EncodedChunk:
    """用 HF tokenizer 把 chunk 的 words 编码为子词序列。

    某些 token 可能被分词器编码为 0 个子词（极少数控制字符或在 do_lower_case
    下被剥离）。这种情况下我们退化为塞入一个 ``[UNK]`` 子词，使 ``first_subword
    _idx`` 与 word 一一对应。
    """
    enc = tokenizer(
        chunk.words,
        is_split_into_words=True,
        truncation=True,
        max_length=max_seq_length,
        return_attention_mask=True,
        add_special_tokens=True,
    )
    input_ids: List[int] = list(enc["input_ids"])
    attn: List[int] = list(enc["attention_mask"])
    word_ids = enc.word_ids()  # type: ignore[attr-defined]

    # 找到每个 word 的第一个子词位置
    first_pos: Dict[int, int] = {}
    for pos, wid in enumerate(word_ids):
        if wid is None:
            continue
        if wid not in first_pos:
            first_pos[wid] = pos

    # 检查截断：若 tokenizer 只放下了前 k 个 word，剩余 word 会缺失
    n_words_present = len(first_pos)
    if n_words_present < len(chunk.words):
        # 截断发生：仅保留前 n_words_present 个 word，让长度对齐
        n_keep = n_words_present
        kept_words = chunk.words[:n_keep]
        kept_labels = chunk.labels[:n_keep] if chunk.labels is not None else None
        kept_first_subword_idx = [first_pos[i] for i in range(n_keep)]
    else:
        # 处理 word 被分词器吃成 0 个子词的退化情况
        unk_id = tokenizer.unk_token_id
        if unk_id is None:
            unk_id = tokenizer.pad_token_id or 0
        kept_words = list(chunk.words)
        kept_labels = list(chunk.labels) if chunk.labels is not None else None
        kept_first_subword_idx = []
        for i in range(len(kept_words)):
            if i in first_pos:
                kept_first_subword_idx.append(first_pos[i])
            else:
                # 在末尾追加一个 UNK 子词作为该 word 的占位
                # 注意：要在 [SEP] 之前插入；简单做法是覆盖最后一个位置
                # （此分支极罕见，不影响整体性能；在末尾插入再确保 max_seq_length）
                if len(input_ids) >= max_seq_length:
                    # 已到上限，没办法再加 token；让该 word 指向 [CLS]，
                    # CRF 会照常预测但精度可能下降（极少数路径）
                    kept_first_subword_idx.append(0)
                else:
                    insert_pos = len(input_ids) - 1  # 在 [SEP] 前
                    if insert_pos < 1:
                        insert_pos = len(input_ids)
                    input_ids.insert(insert_pos, unk_id)
                    attn.insert(insert_pos, 1)
                    kept_first_subword_idx.append(insert_pos)
    return EncodedChunk(
        sentence_idx=chunk.sentence_idx,
        word_start=chunk.word_start,
        n_words=len(kept_words),
        input_ids=input_ids,
        attention_mask=attn,
        first_subword_idx=kept_first_subword_idx,
        labels=kept_labels,
    )


# ----------------------------------------------------------------------
# Dataset / Collator
# ----------------------------------------------------------------------


class NERDataset(Dataset):
    """简单地保存 ``EncodedChunk`` 列表。"""

    def __init__(self, encoded_chunks: List[EncodedChunk]) -> None:
        self.encoded_chunks = encoded_chunks

    def __len__(self) -> int:
        return len(self.encoded_chunks)

    def __getitem__(self, idx: int) -> EncodedChunk:
        return self.encoded_chunks[idx]


def make_collate_fn(pad_token_id: int):
    """返回一个 collate_fn，padding 到 batch 内最长。"""

    def _collate(batch: List[EncodedChunk]) -> Dict[str, torch.Tensor]:
        max_sub = max(len(c.input_ids) for c in batch)
        max_word = max(c.n_words for c in batch)

        B = len(batch)
        input_ids = torch.full((B, max_sub), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((B, max_sub), dtype=torch.long)
        first_subword_idx = torch.zeros((B, max_word), dtype=torch.long)
        word_mask = torch.zeros((B, max_word), dtype=torch.bool)
        labels = torch.zeros((B, max_word), dtype=torch.long)
        sentence_idx = torch.zeros((B,), dtype=torch.long)
        word_start = torch.zeros((B,), dtype=torch.long)
        n_words = torch.zeros((B,), dtype=torch.long)
        has_labels = batch[0].labels is not None

        for i, c in enumerate(batch):
            ls = len(c.input_ids)
            input_ids[i, :ls] = torch.tensor(c.input_ids, dtype=torch.long)
            attention_mask[i, :ls] = torch.tensor(c.attention_mask, dtype=torch.long)
            lw = c.n_words
            first_subword_idx[i, :lw] = torch.tensor(
                c.first_subword_idx, dtype=torch.long
            )
            word_mask[i, :lw] = True
            if has_labels and c.labels is not None:
                labels[i, :lw] = torch.tensor(c.labels, dtype=torch.long)
            sentence_idx[i] = c.sentence_idx
            word_start[i] = c.word_start
            n_words[i] = lw

        out = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "first_subword_idx": first_subword_idx,
            "word_mask": word_mask,
            "sentence_idx": sentence_idx,
            "word_start": word_start,
            "n_words": n_words,
        }
        if has_labels:
            out["labels"] = labels
        return out

    return _collate


# ----------------------------------------------------------------------
# 端到端构建训练 / 验证集
# ----------------------------------------------------------------------


def build_dataset(
    sentences_with_tags: Sequence[Tuple[List[str], List[str]]],
    tokenizer,
    label2id: Dict[str, int],
    max_seq_length: int = 512,
    max_words_per_chunk: int = 200,
    have_labels: bool = True,
) -> NERDataset:
    """从 ``[(tokens, tags), ...]`` 构造 ``NERDataset``。"""
    sentences = [tokens for tokens, _ in sentences_with_tags]
    if have_labels:
        sentence_labels: Optional[List[List[int]]] = []
        for _, tags in sentences_with_tags:
            ids = []
            for t in tags:
                if t not in label2id:
                    # 训练数据中不太可能出现未知标签；为稳健起见映射到 O
                    ids.append(label2id.get("O", 0))
                else:
                    ids.append(label2id[t])
            sentence_labels.append(ids)
    else:
        sentence_labels = None

    chunks = chunk_sentences(
        sentences,
        max_words_per_chunk,
        sentence_labels,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
    )
    encoded = [encode_chunk(c, tokenizer, max_seq_length) for c in chunks]
    _verify_no_dropped_words(chunks, encoded, strict=False)
    return NERDataset(encoded)


def build_inference_dataset(
    sentences: Sequence[List[str]],
    tokenizer,
    max_seq_length: int = 512,
    max_words_per_chunk: int = 200,
) -> NERDataset:
    """推理用：仅有 tokens，无 labels。

    推理时**严格不允许**丢词，否则预测无法对齐验证文件；若发生，
    ``_verify_no_dropped_words(strict=True)`` 会抛出。
    """
    chunks = chunk_sentences(
        sentences,
        max_words_per_chunk,
        sentence_labels=None,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
    )
    encoded = [encode_chunk(c, tokenizer, max_seq_length) for c in chunks]
    _verify_no_dropped_words(chunks, encoded, strict=True)
    return NERDataset(encoded)


def _verify_no_dropped_words(
    chunks: Sequence["Chunk"],
    encoded: Sequence["EncodedChunk"],
    strict: bool,
) -> None:
    """检查 encode 后每个 chunk 的 n_words 与原 chunk 的 word 数一致。"""
    dropped_total = 0
    for c, e in zip(chunks, encoded):
        d = len(c.words) - e.n_words
        if d > 0:
            dropped_total += d
    if dropped_total > 0:
        msg = (
            f"[chunking] 共有 {dropped_total} 个 word 因子词截断被丢弃；"
            "可减小 --max_words_per_chunk 或增大 --max_seq_length。"
        )
        if strict:
            raise ValueError(msg)
        print(msg)


def reassemble_predictions(
    n_sentences: int,
    encoded_chunks: List[EncodedChunk],
    chunk_predictions: List[List[int]],
    id2label: Sequence[str],
) -> List[List[str]]:
    """把 chunk 级预测重新按句子拼回 ``[[tag, tag, ...], ...]``。"""
    if len(chunk_predictions) != len(encoded_chunks):
        raise ValueError(
            f"chunk 数 {len(encoded_chunks)} 与预测数 {len(chunk_predictions)} 不一致"
        )
    # 按 (sentence_idx, word_start) 排序，保证每个句子内 chunks 顺序拼接
    order = sorted(
        range(len(encoded_chunks)),
        key=lambda i: (encoded_chunks[i].sentence_idx, encoded_chunks[i].word_start),
    )
    sent_tags: List[List[str]] = [[] for _ in range(n_sentences)]
    for i in order:
        chunk = encoded_chunks[i]
        preds = chunk_predictions[i]
        if len(preds) != chunk.n_words:
            raise ValueError(
                f"chunk {i} 的预测长度 {len(preds)} 与 n_words {chunk.n_words} 不一致"
            )
        sent_tags[chunk.sentence_idx].extend(id2label[p] for p in preds)
    return sent_tags
