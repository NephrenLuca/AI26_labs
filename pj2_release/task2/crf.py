"""手写线性链 CRF（Conditional Random Field）。

数学约定
--------
对于长度为 T 的输入序列 x 与标签序列 y，定义打分函数

.. math::
    s(x, y) = \\pi_{y_0} + \\sum_{t=0}^{T-1} \\phi(x, t)\\cdot W_{y_t}
              + \\sum_{t=1}^{T-1} A_{y_{t-1}, y_t} + \\eta_{y_{T-1}}

其中：

- :math:`\\phi(x, t)` 为位置 t 的稀疏二值特征指示向量；
- :math:`W \\in \\mathbb{R}^{F\\times K}`：每个特征 × 每个标签的权重；
- :math:`A \\in \\mathbb{R}^{K\\times K}`：标签转移矩阵；
- :math:`\\pi, \\eta \\in \\mathbb{R}^{K}`：起始 / 终止转移得分；
- :math:`F` = 特征数，:math:`K` = 标签数。

条件概率 :math:`P(y|x) = \\exp(s(x,y)) / Z(x)`，其中归一化因子 Z(x)
通过前向算法在对数空间用 logsumexp 计算。

实现细节
--------
- **稀疏发射特征**：用 :class:`torch.nn.EmbeddingBag` (mode='sum')。每个位置上提供
  一个变长的特征 id 列表（拼接成 1D，配合 offsets），输出对应位置的发射打分向量。
  特征 id=0 对应 ``<EMPTY>`` 占位 token，``padding_idx=0`` 让该 id 的贡献始终为 0
  且无梯度。
- **批次掩码**：对 padded 位置不更新前向 / 维特比 alpha；gold 序列打分时按掩码
  乘 0 即可；终止转移 :math:`\\eta` 加到每个样本最后一个真实位置上的标签。
- **数值稳定**：所有累加都在 log 空间用 ``torch.logsumexp``。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn


# ============================ 批数据封装 ====================================


@dataclass
class CRFBatch:
    """一个 mini-batch，已搬到目标 device。"""

    feat_ids: torch.Tensor   # 1D long, 所有位置的特征 id 拼接
    offsets: torch.Tensor    # 1D long, 长度 B*L，每个位置的 bag 起始下标
    tags: torch.Tensor       # (B, L) long
    mask: torch.Tensor       # (B, L) bool
    batch_size: int
    max_len: int

    def to(self, device: torch.device) -> "CRFBatch":
        return CRFBatch(
            feat_ids=self.feat_ids.to(device),
            offsets=self.offsets.to(device),
            tags=self.tags.to(device),
            mask=self.mask.to(device),
            batch_size=self.batch_size,
            max_len=self.max_len,
        )


def collate(
    samples: List[Tuple[List[List[int]], List[int]]],
) -> CRFBatch:
    """将 ``[(feat_id_lists_per_position, tag_id_list)]`` 收集成 :class:`CRFBatch`。

    - 所有空位置（OOV 全过滤）应在上游已被回填为 ``[0]``（``<EMPTY>``）；
    - 序列右侧用 ``<EMPTY>`` 填充至 ``max_len``，并在 ``mask`` 中标记 False；
    - 标签 padded 位置写入 0（任意合法 id 即可，会被 mask 屏蔽）。
    """
    seq_lens = [len(tags) for _, tags in samples]
    B = len(samples)
    max_len = max(seq_lens) if seq_lens else 1

    flat_feats: List[int] = []
    offsets_list: List[int] = []
    tags = torch.zeros(B, max_len, dtype=torch.long)
    mask = torch.zeros(B, max_len, dtype=torch.bool)

    cur = 0
    for b, (feat_pos, tag_seq) in enumerate(samples):
        L = len(tag_seq)
        tags[b, :L] = torch.tensor(tag_seq, dtype=torch.long)
        mask[b, :L] = True
        for t in range(max_len):
            offsets_list.append(cur)
            if t < L:
                ids = feat_pos[t] if feat_pos[t] else [0]
            else:
                ids = [0]  # padding 位置：单独的 <EMPTY> bag
            flat_feats.extend(ids)
            cur += len(ids)

    return CRFBatch(
        feat_ids=torch.tensor(flat_feats, dtype=torch.long),
        offsets=torch.tensor(offsets_list, dtype=torch.long),
        tags=tags,
        mask=mask,
        batch_size=B,
        max_len=max_len,
    )


# ============================ CRF 模型 ====================================


class LinearChainCRF(nn.Module):
    """线性链 CRF：发射分用稀疏特征 EmbeddingBag，转移用全连接矩阵。

    Parameters
    ----------
    num_features : int
        特征词表大小（含 ``<EMPTY>`` 占位）。
    num_tags : int
        标签数。
    """

    def __init__(self, num_features: int, num_tags: int) -> None:
        super().__init__()
        self.num_features = num_features
        self.num_tags = num_tags

        # 稀疏发射：每个特征 -> 各标签的得分；mode='sum' 实现一组特征的指示和。
        self.emit = nn.EmbeddingBag(
            num_features, num_tags, mode="sum", padding_idx=0
        )
        nn.init.zeros_(self.emit.weight)

        # 转移参数。所有项零初始化（CRF 的标准做法）。
        self.transitions = nn.Parameter(torch.zeros(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.zeros(num_tags))
        self.end_transitions = nn.Parameter(torch.zeros(num_tags))

    # --------- 发射分 ---------

    def emit_scores(self, batch: CRFBatch) -> torch.Tensor:
        """返回 (B, L, K) 的发射得分张量。"""
        flat = self.emit(batch.feat_ids, batch.offsets)  # (B*L, K)
        return flat.view(batch.batch_size, batch.max_len, self.num_tags)

    # --------- 前向算法（log Z） ---------

    def _forward_alg(
        self, emit: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """计算 log Z(x)。

        Parameters
        ----------
        emit : (B, L, K) float
        mask : (B, L) bool

        Returns
        -------
        (B,) float, log partition per sample.
        """
        B, L, K = emit.shape
        # alpha[b, j] = log sum over所有以 j 结尾的部分路径分数
        alpha = self.start_transitions.unsqueeze(0) + emit[:, 0]  # (B, K)
        for t in range(1, L):
            # broadcast: (B, K_prev, 1) + (1, K_prev, K_cur) + (B, 1, K_cur)
            inner = (
                alpha.unsqueeze(2)
                + self.transitions.unsqueeze(0)
                + emit[:, t].unsqueeze(1)
            )
            new_alpha = torch.logsumexp(inner, dim=1)  # (B, K)
            mask_t = mask[:, t : t + 1]                # (B, 1)
            alpha = torch.where(mask_t, new_alpha, alpha)
        # 终止转移加在最后一个真实位置（mask 已保证 alpha 停留于此）
        final = alpha + self.end_transitions.unsqueeze(0)
        return torch.logsumexp(final, dim=1)

    # --------- gold 序列打分 ---------

    def _score_sequence(
        self, emit: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """计算 s(x, y) 对每个样本的打分。

        Parameters
        ----------
        emit : (B, L, K)
        tags : (B, L) long
        mask : (B, L) bool

        Returns
        -------
        (B,) float
        """
        B, L, _K = emit.shape
        device = emit.device
        batch_idx = torch.arange(B, device=device)

        # 第 0 步：起始转移 + 发射
        score = self.start_transitions[tags[:, 0]] + emit[batch_idx, 0, tags[:, 0]]

        # t = 1..L-1 累加 transition + emission（按 mask 屏蔽 padding 步）
        for t in range(1, L):
            trans_t = self.transitions[tags[:, t - 1], tags[:, t]]
            emit_t = emit[batch_idx, t, tags[:, t]]
            step = trans_t + emit_t
            score = score + step * mask[:, t].to(score.dtype)

        # 加上终止转移：每个样本最后一个真实位置上的标签
        seq_lens = mask.sum(dim=1)                       # (B,)
        last_idx = (seq_lens - 1).clamp(min=0)
        last_tags = tags.gather(1, last_idx.unsqueeze(1)).squeeze(1)
        score = score + self.end_transitions[last_tags]
        return score

    # --------- 维特比解码 ---------

    def _viterbi_decode(
        self, emit: torch.Tensor, mask: torch.Tensor
    ) -> List[List[int]]:
        """返回每个样本的最优标签序列。"""
        B, L, K = emit.shape
        alpha = self.start_transitions.unsqueeze(0) + emit[:, 0]   # (B, K)
        backpointers: List[torch.Tensor] = []                      # 每项 (B, K) long

        for t in range(1, L):
            # scores[b, prev, cur] = alpha[b, prev] + A[prev, cur]
            scores_t = alpha.unsqueeze(2) + self.transitions.unsqueeze(0)
            best_prev_score, best_prev = scores_t.max(dim=1)       # both (B, K)
            new_alpha = best_prev_score + emit[:, t]
            mask_t = mask[:, t : t + 1]
            alpha = torch.where(mask_t, new_alpha, alpha)
            backpointers.append(best_prev)

        final = alpha + self.end_transitions.unsqueeze(0)          # (B, K)
        seq_lens = mask.sum(dim=1)                                 # (B,)

        # 路径回溯（在 CPU 上做，避免大量 .item() 同步开销过大）
        final_cpu = final.detach().cpu()
        bp_cpu = [bp.detach().cpu() for bp in backpointers]
        seq_lens_cpu = seq_lens.cpu().tolist()

        seqs: List[List[int]] = []
        for b in range(B):
            Lb = seq_lens_cpu[b]
            if Lb <= 0:
                seqs.append([])
                continue
            last = int(final_cpu[b].argmax().item())
            tags_b = [last]
            # backpointers[t-1] 给出"step t 的 best prev"，遍历 t = Lb-1 → 1
            for t in range(Lb - 1, 0, -1):
                last = int(bp_cpu[t - 1][b, last].item())
                tags_b.append(last)
            tags_b.reverse()
            seqs.append(tags_b)
        return seqs

    # --------- 高层接口 ---------

    def neg_log_likelihood(self, batch: CRFBatch) -> torch.Tensor:
        """返回标量损失 ``mean_b [log Z(x_b) - s(x_b, y_b)]``。"""
        emit = self.emit_scores(batch)
        log_z = self._forward_alg(emit, batch.mask)
        gold = self._score_sequence(emit, batch.tags, batch.mask)
        return (log_z - gold).mean()

    @torch.no_grad()
    def decode(self, batch: CRFBatch) -> List[List[int]]:
        """对一个 batch 执行 Viterbi 解码。"""
        emit = self.emit_scores(batch)
        return self._viterbi_decode(emit, batch.mask)
