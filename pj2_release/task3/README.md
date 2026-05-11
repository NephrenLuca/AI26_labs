# 任务三：Transformer + CRF 实现命名实体识别

## 1. 总体架构

```
                 ┌────────────────────────────┐
   tokens   →    │   BERT (huggingface)       │  →  子词级 hidden states
                 └────────────────────────────┘
                              │ gather (每个 word 的第 1 个子词)
                              ▼
                 ┌────────────────────────────┐
                 │  (可选) BiLSTM             │
                 └────────────────────────────┘
                              │ Linear
                              ▼
                          word-level logits   (B, T_word, num_tags)
                              │
                              ▼
                 ┌────────────────────────────┐
                 │   线性链 CRF（手写）       │  →  loss / Viterbi 序列
                 └────────────────────────────┘
```

* **Transformer 部分**：用 HuggingFace `transformers` 加载预训练 BERT，
  允许使用任何 BERT 系列模型（中文 `bert-base-chinese`、英文
  `bert-base-cased`，或更大的 RoBERTa/MacBERT 等同结构模型）。
* **CRF 部分（手写）**：见 [`crf.py`](crf.py)。包含
  - 路径得分计算 `_compute_score`
  - 前向算法 log 配分函数 `_compute_log_partition`（log-sum-exp）
  - Viterbi 解码 `_viterbi_decode`
  - 负对数似然 NLL，作为训练 loss
  纯 PyTorch 实现，未调用 `torchcrf`/`pytorch-crf` 等任何 CRF 第三方库。

## 2. 目录结构

```
task3/
├── crf.py            # 手写 CRF 层（forward / Viterbi / NLL）
├── data_utils.py     # CoNLL 读写 + 子词对齐 + Dataset / Collator
├── model.py          # BertCRFForNER（BERT + 可选 BiLSTM + CRF）
├── evaluate.py       # 与 NER/check.py 等价的评测，含 micro_f1
├── run.py            # 主入口（训练、预测、评测、CLI）
├── requirements.txt
└── README.md
```

输出位置：`task3/outputs/{Chinese,English}_pred.txt`，与样例
`NER/example_data/example_my_result.txt` 行结构一致；空行保留，
非空行为 `token<space>预测标签`。

## 3. 安装

```bash
pip install -r requirements.txt
```

`torch` 必须根据本机 CUDA 版本单独安装（见 `requirements.txt` 内注释）。
评测脚本依赖只需要 `scikit-learn`。

## 4. 预训练模型下载

代码默认按语言自动选择模型：

| 语言    | 模型                     | 主页                                                   |
| ------- | ------------------------ | ------------------------------------------------------ |
| 中文    | `bert-base-chinese`      | https://huggingface.co/bert-base-chinese               |
| 英文    | `bert-base-cased`        | https://huggingface.co/google-bert/bert-base-cased     |

### 4.1 直接联网拉取（默认）

第一次运行 `run.py` 时，`transformers` 会自动从 HuggingFace 下载模型并缓存
到 `~/.cache/huggingface/`，之后离线可用。

### 4.2 国内镜像（如直连不通）

```bash
export HF_ENDPOINT=https://hf-mirror.com
python run.py --language Chinese --device cuda --gpu 0
```

或预下载到本地（推荐用于无外网的训练机）：

```bash
# 在能联网的机器上执行
pip install huggingface_hub
huggingface-cli download bert-base-chinese          --local-dir ./pretrained/bert-base-chinese
huggingface-cli download google-bert/bert-base-cased --local-dir ./pretrained/bert-base-cased

# 然后传输到训练机，运行时显式指定本地路径
python run.py --language Chinese --model_name_or_path ./pretrained/bert-base-chinese
python run.py --language English --model_name_or_path ./pretrained/bert-base-cased
```

也可以用更强的预训练模型（同样的接口即可）：

| 语言 | 推荐替代                                | 说明                          |
| ---- | --------------------------------------- | ----------------------------- |
| 中文 | `hfl/chinese-roberta-wwm-ext`           | 全词掩码 RoBERTa，简历 NER 常用 |
| 中文 | `hfl/chinese-macbert-base`              | MacBERT，效果略好             |
| 英文 | `roberta-base` / `roberta-large`        | 替换为 RoBERTa 时 tokenizer 仍兼容 `is_split_into_words` |

通过 `--model_name_or_path` 指定即可，无需改代码。

## 5. 设备 & 多卡

```text
--device {cpu,cuda}     选择运行设备
--gpu    "0" / "0,1,2"  仅 cuda 时生效；指定可见物理 GPU id（0-7）
```

行为：

* `--device cpu`：纯 CPU 运行，适合调试。
* `--device cuda --gpu 0`：使用单卡（物理 GPU 0）。
* `--device cuda --gpu 0,3`：使用 GPU 0 与 GPU 3，自动启用
  `torch.nn.DataParallel`（model 在 `cuda:0` 上 broadcast，反向自动 reduce）。

> 实现方式：在 `import torch` 之前先设置
> `os.environ["CUDA_VISIBLE_DEVICES"] = "0,3"`，PyTorch 内部会把这两块卡映
> 射成 `cuda:0` / `cuda:1`，模型放到 `cuda:0` 即可。这与服务器上的 0-7 物
> 理卡空间一一对应，互不冲突。

## 6. 运行示例

```bash
# 1) 中英文一键训练 + 预测 + 评测，单卡
python run.py --language all --device cuda --gpu 0 --epochs 3

# 2) 仅训练中文，4 卡 DataParallel，更大 batch
python run.py --language Chinese --device cuda --gpu 0,1,2,3 \
              --batch_size 64 --eval_batch_size 128 --epochs 5

# 3) CPU 烟雾测试
python run.py --language English --device cpu \
              --epochs 1 --batch_size 4 --eval_batch_size 8

# 4) 切换到 RoBERTa-WWM-Ext (中文常用更强模型)
python run.py --language Chinese --device cuda --gpu 0 \
              --model_name_or_path hfl/chinese-roberta-wwm-ext --epochs 5

# 5) 加一层 BiLSTM
python run.py --language all --device cuda --gpu 0 --use_bilstm
```

每个 epoch 末会做一次 Viterbi 解码 + 评测，自动保留 dev micro-F1 最高的预测
作为 `outputs/{lang}_pred.txt`；中间也会写一份 `outputs/{lang}_pred_best.txt`
作为 best 副本。

加 `--save_model` 会同时把最佳 checkpoint 保存为 `outputs/{lang}_best.pt`。

## 7. CRF 原理速记（面试要点）

* **建模目标**：在给定观测序列 \(x\) 下，对标签序列 \(y\) 建模条件概率
  \[
    p(y \mid x) = \frac{1}{Z(x)} \exp\Big( \sum_t \phi_t(y_t, x) + \sum_t \psi(y_{t-1}, y_t) \Big)
  \]
  其中 \(\phi_t\) 来自 BERT 的发射分（`emissions`），\(\psi\) 是可学习的转移
  矩阵。归一化项
  \[ Z(x) = \sum_{y'} \exp( \cdot ) \]
  通过 **前向算法** 在 log 空间用 `logsumexp` 计算，时间复杂度 \(O(T K^2)\)。
* **训练**：最大化 \(\log p(y \mid x)\)，等价于最小化 NLL =
  \(\log Z(x) - \mathrm{score}(x, y)\)。
* **推理**：用 **Viterbi** 在 log 空间做 max-product，回溯 backpointer 得到
  最优标签序列。
* **与本任务的关系**：BERT 给出每个位置的标签发射分，但缺少对相邻标签合法
  性的约束（如 `B-PER` 后只能跟 `I-PER` 或 `O`）；CRF 把这种结构性偏置以
  转移矩阵的形式学进来，能显著降低 “B-X` 后接 `I-Y`” 这类不合法预测。

## 8. 与样例输出格式的一致性

* `evaluate.py` 与 `NER/check.py` 完全等价（同样 `sklearn.metrics.classification_
  report`，同样的 `sorted_labels`），仅显式 UTF-8 打开文件，避免 Windows
  GBK 解码问题；Linux 服务器上结果一致。
* `data_utils.write_predictions` 严格保留原 validation 文件的空行结构，
  仅把每个非空行的末尾标签替换为模型预测；行数、空行位置与样例完全一致。
