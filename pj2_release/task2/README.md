# 任务二：CRF 实现命名实体识别

手写线性链 CRF + CRF++-style 离散特征 + PyTorch 训练，在中文 / 英文两个数据集上跑 NER。

> 设计取舍：任务三（Transformer + CRF）用神经编码器代替手工特征，因此本任务保留**离散稀疏特征**的传统 CRF 形态，便于在面试中讲清 CRF 原理本身（forward / Viterbi / 配分函数）。

---

## 1. 原理

### 线性链 CRF 的概率模型

给定输入序列 $x = (x_0, \dots, x_{T-1})$ 与标签序列 $y = (y_0, \dots, y_{T-1})$，定义打分函数

$$
s(x, y) = \pi_{y_0} + \sum_{t=0}^{T-1} \phi(x, t)\!\cdot\! W_{:, y_t} + \sum_{t=1}^{T-1} A_{y_{t-1}, y_t} + \eta_{y_{T-1}}
$$

- $\phi(x, t) \in \{0, 1\}^{F}$：位置 $t$ 处的稀疏二值特征指示向量；
- $W \in \mathbb{R}^{F\times K}$：每个特征 × 每个标签的权重矩阵；
- $A \in \mathbb{R}^{K\times K}$：标签转移矩阵；$\pi, \eta$：起始 / 终止转移得分；
- $F$ 为特征数，$K$ 为标签数。

条件概率：

$$
P(y\mid x) = \frac{\exp\big(s(x, y)\big)}{Z(x)},\qquad Z(x) = \sum_{y'} \exp\big(s(x, y')\big).
$$

### 训练目标（极大化条件似然）

负对数似然

$$
\mathcal{L} = -\log P(y_{\text{gold}}\mid x) = \log Z(x) - s(x, y_{\text{gold}}).
$$

`log Z(x)` 用**前向算法**在 log 空间通过 `logsumexp` 计算（$\mathcal{O}(T K^2)$）：

$$
\alpha_0(j) = \pi_j + \phi(x, 0)\!\cdot\! W_{:, j},\quad
\alpha_t(j) = \phi(x, t)\!\cdot\! W_{:, j} + \operatorname{logsumexp}_{i}\big(\alpha_{t-1}(i) + A_{i, j}\big).
$$

### 推理（解码）

最大概率序列等价于求最大 $s(x, y)$ —— 用**Viterbi 算法**：把 `logsumexp` 换成 `max` 并记录回溯指针。

### 与 HMM 的本质差异

| | HMM (任务一) | CRF (本任务) |
| --- | --- | --- |
| 模型 | 生成式 $P(x, y)$ | 判别式 $P(y\mid x)$ |
| 发射概率 | 必须满足 $\sum_x P(x|y) = 1$ | 直接学 $\phi(x,t)\cdot W_{:, y_t}$，无需归一化 |
| 特征 | 单 token | 任意上下文 / 形态学 / 模板组合 |
| 标签独立性假设 | $P(y_t | y_{t-1})$ 假设 | 仅做马尔可夫假设，对 $x$ 全局观察 |

---

## 2. 实现

### 2.1 文件结构

```
task2/
├── data_utils.py   # CoNLL 数据读写（与 task1 对齐）
├── features.py     # CRF++ 模板特征 + 形态学（shape / 前后缀）
├── crf.py          # LinearChainCRF（手写 forward / Viterbi / score_sequence）
├── evaluate.py     # 与 NER/check.py 等价的评测函数
├── run.py          # 主入口：argparse + 训练 + 解码 + 评测
├── requirements.txt
└── outputs/        # 运行后生成：{Lang}_pred.txt、可选的 {Lang}_crf.pt
```

### 2.2 特征模板

参考 `NER/template_for_crf.utf8`，对每个位置 $t$ 提取：

| 类别 | 模板 | 说明 |
| --- | --- | --- |
| 单 token | `U00..U04` | $x_{t-2}, x_{t-1}, x_t, x_{t+1}, x_{t+2}$ |
| 二元 token | `U05..U09` | 5 组相邻 / 跳一 token 的拼接 |
| 形态学 | `S01..S03` | $t-1, t, t+1$ 的 word shape（大小写 / 数字 / 后缀类别） |
| 小写化 | `LOW` | （仅英文）$x_t$ 的 lowercase 形 |
| 前/后缀 | `PRE2/3, SUF2/3` | （仅英文）$x_t$ 的前 2-3 字符 / 后 2-3 字符 |

边界外位置用 `<BOS-2> / <BOS-1> / <EOS+1> / <EOS+2>` 占位。

每个特征字符串形如 `"U02=Bradford"`、`"S02=INITCAP"`、`"PRE3=mar"`。训练时统计频次，频次 $<$ `--feat_min_count` 的特征被丢弃（L0 正则，显著缩减词表、缓解过拟合）。

### 2.3 模型实现

发射部分用 `nn.EmbeddingBag(num_features, num_tags, mode='sum')`：每个位置上一组激活特征 id 的"和"恰好等于 $\phi(x, t) \cdot W$。这样既保持稀疏性，又能在 GPU 上高效批处理。

转移部分是 `(K, K)` 的全连接矩阵；起始 / 终止转移各一个 `(K,)` 向量。

`LinearChainCRF` 类包含三个核心方法：

- `_forward_alg`：前向算法，向量化的 logsumexp 递推。
- `_score_sequence`：对给定 gold 序列计算 $s(x, y)$，按 mask 正确处理 padding。
- `_viterbi_decode`：维特比解码并回溯路径。

NLL 损失 = `_forward_alg(...) - _score_sequence(...)` 的均值。

### 2.4 训练

- AdamW + 梯度裁剪
- 全部参数零初始化（CRF 标准做法）
- `padding_idx=0` 让 `<EMPTY>` 占位特征始终零贡献且无梯度

---

## 3. 使用方法

### 3.1 安装依赖

```bash
pip install -r requirements.txt
```

CPU-only 环境下直接 `pip install torch` 即可；GPU 服务器请按 CUDA 版本安装匹配 wheel：

```bash
# 例：CUDA 12.x
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 3.2 运行

```bash
# 中英文都跑（CPU）
python run.py --language all

# 在 0 号 GPU 上训练英文
python run.py --language English --device cuda --gpu 3 --epochs 15

# 限制可见为 GPU 0、2，模型占用第一张可见卡
python run.py --language Chinese --device cuda --gpu 0,2 --batch_size 64

# 关闭形态学特征做消融
python run.py --language English --no_shape --no_affix
```

### 3.3 主要 CLI

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--language` | `all` | `Chinese` / `English` / `all` |
| `--device` | `cpu` | `cpu` / `cuda` |
| `--gpu` | `0` | `--device cuda` 下指定的物理卡 id，可逗号分隔（通过 `CUDA_VISIBLE_DEVICES` 限制可见） |
| `--epochs` | 15 | 训练轮数 |
| `--batch_size` | 32 | mini-batch 句数 |
| `--lr` | 1e-2 | AdamW 学习率 |
| `--weight_decay` | 1e-5 | L2 正则 |
| `--feat_min_count` | 2 | 训练集中频次 $<$ 该阈值的特征被丢弃 |
| `--max_grad_norm` | 5.0 | 梯度裁剪阈值 |
| `--no_shape` | off | 关闭 word shape / 大小写 / 后缀类别特征 |
| `--no_affix` | off | 关闭 PREn/SUFn 字符级前后缀特征 |
| `--save_model` | off | 把权重 + 词表保存到 `outputs/{Lang}_crf.pt` |

### 3.4 输出

- `outputs/{Chinese|English}_pred.txt`：与 `validation.txt` 行格式一致的预测文件，`check.py` 可直接消费。
- 控制台打印 `sklearn.metrics.classification_report`（与 `NER/check.py` 完全等价；`micro avg F1-score` 即为评测指标）。

---

## 4. 关键决策回顾（面试备忘）

1. **为什么写离散特征 CRF 而非 BiLSTM-CRF？** 任务三专门要求 Transformer + CRF，本任务保持"经典 CRF"原貌，让 CRF 部件本身（前向算法、配分函数、Viterbi）是讲解重点。
2. **为什么用 `EmbeddingBag` 而不是稀疏矩阵？** 同等数学等价（mode='sum' 等于稀疏 0/1 向量乘权重矩阵），且 PyTorch 内置 GPU 实现，省去手写稀疏矩阵 backward。
3. **特征频次阈值的作用？** 离散特征模型的特征数会随语料增长很快达 $10^6$ 量级；阈值 = 2 可裁掉一半以上"只见过一次"的噪声特征，模型小、收敛快、泛化好。
4. **`padding_idx=0`？** 对应 `<EMPTY>` 占位特征：当某个位置上所有特征都是 OOV 时回填它，保证 `EmbeddingBag` 不会遇到空 bag；同时该 id 的权重始终零、永不更新。
5. **掩码处理？** 前向递推用 `torch.where` 保持 padded 位置 alpha 不变；gold 打分用 mask 把 padded 步的贡献乘 0；Viterbi 在每个样本各自的 `seq_lens` 处取 argmax 并回溯。
6. **设备控制？** 在 `import torch` 之前设置 `CUDA_VISIBLE_DEVICES`，能精确选择物理 GPU；`--device cpu` 时完全跳过 cuda 检测。
