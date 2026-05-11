"""手写线性链 CRF（Linear-chain Conditional Random Field）。

接口与训练目标
--------------
对于序列 :math:`x_{1:T}` 与标签 :math:`y_{1:T}`，CRF 用三组参数描述一条序列的得分：

* ``start_transitions``  : :math:`s_0(y_1)`，开始 → 第一个标签的得分；
* ``end_transitions``    : :math:`s_E(y_T)`，最后一个有效标签 → 结束的得分；
* ``transitions``        : :math:`s(y_{t-1}, y_t)`，相邻标签的转移得分。

每条序列的得分定义为 ``发射分 + 转移分 + 起止分``。

* **训练**：最大化 ``log p(y|x) = score(x, y) - log Z(x)``，其中 ``log Z`` 通过
  前向算法在 log 空间用 ``logsumexp`` 计算。``forward`` 返回 mini-batch 上的
  平均 NLL，可直接 ``loss.backward()``。
* **解码**：``decode`` 用 Viterbi 给每条样本返回最优标签序列，长度 = 该样本
  的有效 mask 长度。

向量化
------
所有时间步对 batch 维与标签维都做了张量化，实现是纯 PyTorch；时间维上仍用
显式 ``for t``（这是 CRF 的固有依赖，无法绕过），但对单步内的运算完全 GPU
并行；典型 1B-LSTM/Transformer 后接 CRF 的序列长度 ~50-200，速度足够。

约定
----
* 输入 ``emissions`` shape = ``(B, T, K)``；``K`` = ``num_tags``。
* ``tags`` shape = ``(B, T)``，长整型，pad 位置任意（被 mask 掩去）。
* ``mask`` shape = ``(B, T)``，bool 或 0/1 浮点；** 必须保证每条样本的有效
  位置位于序列前缀** (即 mask 形如 ``[1,1,...,1,0,...,0]``)；同时 ``mask[:, 0]``
  全部为 1（每条样本至少 1 个 token）。

NLL 的标准推导见 Lafferty et al. 2001，《Conditional Random Fields》。
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn


class CRF(nn.Module):
    """线性链 CRF 解码层，纯手写 forward / Viterbi / NLL。"""

    def __init__(self, num_tags: int) -> None:
        super().__init__()
        if num_tags <= 1:
            raise ValueError(f"num_tags 必须 >= 2，得到 {num_tags}")
        self.num_tags = num_tags
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    @staticmethod
    def _check_inputs(
        emissions: torch.Tensor,
        mask: Optional[torch.Tensor],
        tags: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if emissions.dim() != 3:
            raise ValueError(f"emissions 必须是 3D (B,T,K)，得到 {emissions.shape}")
        B, T, _ = emissions.shape
        if mask is None:
            mask = emissions.new_ones((B, T), dtype=torch.bool)
        if mask.dtype != torch.bool:
            mask = mask.bool()
        if mask.shape != (B, T):
            raise ValueError(f"mask 形状 {mask.shape} 与 emissions 不匹配")
        if not torch.all(mask[:, 0]):
            raise ValueError("mask[:, 0] 必须全为 True (每条样本至少 1 个 token)")
        if tags is not None and tags.shape != (B, T):
            raise ValueError(f"tags 形状 {tags.shape} 与 emissions 不匹配")
        return mask

    # ------------------------------------------------------------------
    # 数学三件套：路径得分、log 配分、Viterbi
    # ------------------------------------------------------------------

    def _compute_score(
        self,
        emissions: torch.Tensor,  # (B, T, K)
        tags: torch.Tensor,  # (B, T) long
        mask: torch.Tensor,  # (B, T) bool
    ) -> torch.Tensor:
        """对给定真实标签序列计算未归一化得分。返回形状 ``(B,)``。"""
        B, T, _ = emissions.shape
        m = mask.float()  # (B, T)
        # 第一时间步：start + emission(0)
        score = self.start_transitions[tags[:, 0]]  # (B,)
        score = score + emissions.gather(2, tags.unsqueeze(2)).squeeze(2)[:, 0]
        # 之后的时间步
        for t in range(1, T):
            trans = self.transitions[tags[:, t - 1], tags[:, t]]  # (B,)
            emit = emissions[:, t].gather(1, tags[:, t].unsqueeze(1)).squeeze(1)
            score = score + (trans + emit) * m[:, t]
        # 最后一个有效时间步索引
        seq_lengths = m.long().sum(dim=1)  # (B,)
        last_idx = (seq_lengths - 1).clamp(min=0)  # (B,)
        last_tags = tags.gather(1, last_idx.unsqueeze(1)).squeeze(1)  # (B,)
        score = score + self.end_transitions[last_tags]
        return score

    def _compute_log_partition(
        self,
        emissions: torch.Tensor,  # (B, T, K)
        mask: torch.Tensor,  # (B, T) bool
    ) -> torch.Tensor:
        """前向算法计算 ``log Z(x)``。返回形状 ``(B,)``。"""
        B, T, K = emissions.shape
        m = mask.float()
        # alpha[b, k] = log( sum_{paths ending at tag k at time t} exp(score) )
        alpha = self.start_transitions.unsqueeze(0) + emissions[:, 0]  # (B, K)
        for t in range(1, T):
            broadcast_alpha = alpha.unsqueeze(2)  # (B, K_prev, 1)
            broadcast_emit = emissions[:, t].unsqueeze(1)  # (B, 1, K_curr)
            # transitions: (K_prev, K_curr)
            new_alpha = torch.logsumexp(
                broadcast_alpha + self.transitions + broadcast_emit, dim=1
            )  # (B, K_curr)
            mask_t = m[:, t].unsqueeze(1)  # (B, 1)
            alpha = mask_t * new_alpha + (1.0 - mask_t) * alpha
        alpha = alpha + self.end_transitions.unsqueeze(0)
        return torch.logsumexp(alpha, dim=1)  # (B,)

    def _viterbi_decode(
        self,
        emissions: torch.Tensor,  # (B, T, K)
        mask: torch.Tensor,  # (B, T) bool
    ) -> List[List[int]]:
        """Viterbi 解码：每条样本返回长度 = 有效 mask 数的 tag id 列表。"""
        B, T, _ = emissions.shape
        m = mask.float()
        # score[b, k] = 当前时间步以 tag=k 结尾的最优路径得分
        score = self.start_transitions.unsqueeze(0) + emissions[:, 0]  # (B, K)
        history: List[torch.Tensor] = []
        for t in range(1, T):
            # broadcast: (B, K_prev, 1) + (K_prev, K_curr) -> (B, K_prev, K_curr)
            scored = score.unsqueeze(2) + self.transitions
            best_score, best_prev = scored.max(dim=1)  # (B, K_curr) each
            new_score = best_score + emissions[:, t]
            mask_t = m[:, t].unsqueeze(1)
            score = mask_t * new_score + (1.0 - mask_t) * score
            history.append(best_prev)  # (B, K)
        score = score + self.end_transitions.unsqueeze(0)  # (B, K)
        seq_lengths = mask.long().sum(dim=1)  # (B,)
        # 对每条样本回溯
        best_tags_all: List[List[int]] = []
        best_last_tags = score.argmax(dim=1)  # (B,)
        for b in range(B):
            length = int(seq_lengths[b].item())
            if length == 0:
                best_tags_all.append([])
                continue
            tags_b: List[int] = [int(best_last_tags[b].item())]
            # history[t-1] 是从 t-1 -> t 的 backpointer；
            # 对于样本长度 L，回溯使用 history[0..L-2]
            for t in range(length - 2, -1, -1):
                tags_b.append(int(history[t][b, tags_b[-1]].item()))
            tags_b.reverse()
            best_tags_all.append(tags_b)
        return best_tags_all

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------

    def forward(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """返回 batch 上的 NLL（默认按样本平均）。

        ``reduction`` 取 ``"mean"`` / ``"sum"`` / ``"token_mean"``：
        - ``mean`` 按样本平均（与多数 CRF 实现一致）
        - ``sum``  返回总和
        - ``token_mean`` 按总有效 token 数平均（与 cross-entropy 行为一致）
        """
        mask = self._check_inputs(emissions, mask, tags)
        gold_score = self._compute_score(emissions, tags, mask)
        log_partition = self._compute_log_partition(emissions, mask)
        nll = log_partition - gold_score  # (B,)
        if reduction == "mean":
            return nll.mean()
        if reduction == "sum":
            return nll.sum()
        if reduction == "token_mean":
            n_tokens = mask.float().sum().clamp(min=1.0)
            return nll.sum() / n_tokens
        raise ValueError(f"未知 reduction: {reduction}")

    def decode(
        self,
        emissions: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> List[List[int]]:
        mask = self._check_inputs(emissions, mask)
        return self._viterbi_decode(emissions, mask)
