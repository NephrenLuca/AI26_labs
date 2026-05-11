"""Transformer (BERT) + 可选 BiLSTM + 手写 CRF 的 NER 模型。

输入约定（与 ``data_utils.make_collate_fn`` 严格一致）
------------------------------------------------------
* ``input_ids``         : (B, T_sub)
* ``attention_mask``    : (B, T_sub)，BERT 的 padding mask
* ``first_subword_idx`` : (B, T_word)，每个原 word 的第一个子词在 ``input_ids`` 中
                          的位置；pad 位置为 0
* ``word_mask``         : (B, T_word) bool，标记真实 word 的位置（CRF 用）
* ``labels``            : (B, T_word) long，仅训练时提供

forward 行为
------------
* 训练 (``labels`` 给定)：返回 ``loss``（标量）
* 推理 (``labels=None``)：返回每条样本的 Viterbi 解码结果 ``List[List[int]]``

实现细节
--------
* 用 ``torch.gather`` 把 BERT 的子词级隐藏向量收集到 word 级，再过 dropout +
  线性投影得到发射分；
* CRF 层完全由 ``crf.CRF`` 提供（手写实现）；
* 顶层支持额外一层 BiLSTM（``--use_bilstm``）。BiLSTM 在 word 级运行，对
  Transformer 的输出做进一步序列建模；论文与实践中常带来小幅提升。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from crf import CRF


class BertCRFForNER(nn.Module):
    """BERT (encoder) → [可选 BiLSTM] → 线性 → 手写 CRF。"""

    def __init__(
        self,
        bert_model: nn.Module,
        num_tags: int,
        hidden_dropout_prob: float = 0.1,
        use_bilstm: bool = False,
        bilstm_hidden: int = 256,
        bilstm_layers: int = 1,
        bilstm_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.bert = bert_model
        hidden_size = self._infer_hidden_size(bert_model)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.use_bilstm = use_bilstm
        if use_bilstm:
            self.bilstm = nn.LSTM(
                input_size=hidden_size,
                hidden_size=bilstm_hidden,
                num_layers=bilstm_layers,
                bidirectional=True,
                batch_first=True,
                dropout=bilstm_dropout if bilstm_layers > 1 else 0.0,
            )
            classifier_in = bilstm_hidden * 2
        else:
            self.bilstm = None
            classifier_in = hidden_size
        self.classifier = nn.Linear(classifier_in, num_tags)
        self.crf = CRF(num_tags)
        self.num_tags = num_tags

    @staticmethod
    def _infer_hidden_size(bert_model: nn.Module) -> int:
        cfg = getattr(bert_model, "config", None)
        if cfg is not None and hasattr(cfg, "hidden_size"):
            return int(cfg.hidden_size)
        # 兜底
        for name, p in bert_model.named_parameters():
            if name.endswith("embeddings.word_embeddings.weight"):
                return p.shape[1]
        raise ValueError("无法从 bert_model 推断 hidden_size")

    # ------------------------------------------------------------------
    # 子词 -> 词级特征
    # ------------------------------------------------------------------

    @staticmethod
    def _gather_word_features(
        sub_hidden: torch.Tensor,  # (B, T_sub, H)
        first_subword_idx: torch.Tensor,  # (B, T_word) long
    ) -> torch.Tensor:
        B, T_word = first_subword_idx.shape
        H = sub_hidden.shape[-1]
        idx = first_subword_idx.unsqueeze(-1).expand(B, T_word, H)
        return sub_hidden.gather(dim=1, index=idx)  # (B, T_word, H)

    # ------------------------------------------------------------------
    # 计算 emissions
    # ------------------------------------------------------------------

    def compute_emissions(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        first_subword_idx: torch.Tensor,
        word_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        sub_hidden = outputs.last_hidden_state  # (B, T_sub, H)
        word_hidden = self._gather_word_features(sub_hidden, first_subword_idx)
        word_hidden = self.dropout(word_hidden)
        if self.bilstm is not None:
            # 用 word_mask 给 LSTM 提供长度，避免 pad 位置参与 BiLSTM
            lengths = word_mask.long().sum(dim=1).clamp(min=1)
            packed = nn.utils.rnn.pack_padded_sequence(
                word_hidden, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            lstm_out, _ = self.bilstm(packed)
            word_hidden, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out, batch_first=True, total_length=word_hidden.shape[1]
            )
            word_hidden = self.dropout(word_hidden)
        emissions = self.classifier(word_hidden)  # (B, T_word, num_tags)
        return emissions

    # ------------------------------------------------------------------
    # 训练 / 推理
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        first_subword_idx: torch.Tensor,
        word_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **_: Dict,
    ) -> Union[torch.Tensor, List[List[int]]]:
        emissions = self.compute_emissions(
            input_ids, attention_mask, first_subword_idx, word_mask
        )
        if labels is not None:
            loss = self.crf(emissions, labels, mask=word_mask, reduction="mean")
            return loss
        return self.crf.decode(emissions, mask=word_mask)
