# 任务一：HMM 命名实体识别

手写隐马尔可夫模型 (HMM) 完成中文 / 英文 NER 任务。**未使用任何机器学习框架**，
训练即"频次统计 + 平滑"；数值后端在 CPU(`numpy`) 与 GPU(`cupy`) 间无缝切换。

支持的模型变种：

- 一阶 / 二阶（trigram）HMM，二阶用 Brants (1999) deleted-interpolation 自动估计 λ
- **TnT 后缀树 OOV (Brants 2000)**：低频词在递归平滑的后缀树上估计
  `p(t | suffix_k)`，按贝叶斯翻成相对发射后**与固定形态类别 (`<INITCAP>` / `<NUM>`
  / …) 的发射 MLE 软合成**
- **shape vocab**：训练时把所有词的"主"shape (最长合规后缀 `<SUF$branch$k$suf>`
  或固定类别 `<CAT$...>`) 都作为伪 vocab 词条；推理时 OOV 走 shape 列
- **结构化转移约束**：自动识别 BIO / BMES，禁止 `O→I-X`、`B-X→I-Y(Y≠X)`、
  `B-X→M-Y(Y≠X)` 等结构上不可能的 tag 跳变
- π / A / B 三套独立 Laplace α，可手动指定或自动 grid-search

---

## 目录结构

```
task1/
├── data_utils.py     # CoNLL 数据读写（保留空行）
├── features.py       # 词形函数 (English / Chinese) + 语言检测
├── suffix_model.py   # SuffixShapeModel + TnTSuffixTree (Brants 2000)
├── constraints.py    # BIO/BMES 转移与起始的 -inf 掩码
├── hmm.py            # HMM 主体：order=1/2，numpy/cupy 双后端
├── evaluate.py       # 与 NER/check.py 等价的跨平台评测
├── run.py            # 主入口：训练 / 网格搜索 / 预测 / 评测
├── requirements.txt
├── README.md
└── outputs/          # Chinese_pred.txt / English_pred.txt
```

---

## 安装

```bash
pip install -r requirements.txt        # CPU 即可跑
# GPU 服务器额外装匹配 CUDA 的 CuPy
pip install cupy-cuda12x                # 或 cupy-cuda11x
```

仅在 `--device cuda` 时才会 import cupy；CPU 机器无需安装。

---

## 运行

```bash
# 推荐：默认即开 TnT 后缀树 + order=2 + 结构约束
python run.py --language all --order 2

# 单语言 + 网格搜索（α、weight_tnt、suffix_min_count）
python run.py --language English --order 2 --tune

# GPU 加速（指定物理 GPU 3）
python run.py --language all --device cuda --gpu 3 --order 2

# 多 GPU 可见性（HMM 只用第一块）
python run.py --language all --device cuda --gpu 0,1,2 --order 2

# 消融
python run.py --language English --order 2 --no_tnt_tree           # 关 TnT
python run.py --language English --order 2 --no_constraints        # 关结构约束
python run.py --language English --order 2 --weight_tnt 0          # 等价于关 TnT
```

### 参数速查

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--language` | `all` | `Chinese` / `English` / `all` |
| `--device` | `cpu` | `cpu` (numpy) / `cuda` (cupy) |
| `--gpu` | `0` | 仅 cuda 时生效，等价于 `CUDA_VISIBLE_DEVICES`，可逗号分隔 |
| `--order` | `2` | HMM 阶数：`1` = bigram，`2` = trigram |
| `--alpha_pi` | `1.0` | π 的 Laplace α |
| `--alpha_trans` | `0.01` | A 的 Laplace α |
| `--alpha_emit` | `1e-4` | B 的 Laplace α |
| `--rare_threshold` | `10` | freq ≤ 该阈值的词参与 TnT 后缀树训练 |
| `--max_suffix_len` | `10` | TnT 后缀最大长度（中文自动 = 1） |
| `--suffix_min_count` | `200` | 后缀计数 ≥ 该阈值才进 SUF shape vocab |
| `--theta_tnt` | `1.0` | TnT 递归平滑权重 |
| `--weight_tnt` | `0.3` | OOV 推理时 TnT 相对发射的混合权重，0 = 关 TnT |
| `--no_tnt_tree` | off | 关闭 TnT 后缀树 |
| `--no_suffix_oov` | off | 关闭整个 shape vocab |
| `--no_constraints` | off | 关闭结构化约束 |
| `--tune` | off | 在 train 上切 mini-dev 做 5D 网格搜索 |
| `--dev_ratio` | `0.1` | 网格搜索的 dev 比例 |
| `--seed` | `0` | 切 mini-dev 的随机种子 |
| `--alpha_*_grid` / `--weight_tnt_grid` / `--suffix_min_count_grid` | 见 `run.py` | 自定义网格 |
| `--batch_size` | `64` | 批量 Viterbi 的 batch |
| `--save_model` | off | 保存模型到 `outputs/<lang>_hmm.pkl` |

---

## 模型说明

### 一阶 HMM (`--order 1`)

经典 bigram tag model：

\[
\hat y = \arg\max_{y_{1:L}}\;
\log\pi(y_1) + \sum_{i=1}^{L} \log B(x_i \mid y_i) + \sum_{i=2}^{L} \log A(y_i \mid y_{i-1})
\]

### 二阶 HMM (`--order 2`)

trigram tag model：

\[
P(y_i \mid y_{i-2}, y_{i-1}) = \lambda_3 P_3(y_i\mid y_{i-2},y_{i-1})
+ \lambda_2 P_2(y_i\mid y_{i-1}) + \lambda_1 P_1(y_i)
\]

权重 \( (\lambda_1,\lambda_2,\lambda_3) \) 由 Brants (1999)
**deleted-interpolation** 自动估计。Viterbi 状态 = `(prev_tag, cur_tag)`，
复杂度 \( O(L T^3) \)。

### 形态特征 OOV (`--no_suffix_oov` 关闭)

训练时所有词的"主"shape 都进入 vocab：

- 优先使用最长的 **TnT 后缀** `<SUF$branch$k$suf>`（出现次数 ≥
  `suffix_min_count` 才算合规）
- 否则回退固定类别 `<CAT$...>`：`<CAT$<INITCAP>>` / `<CAT$<NUM>>` / `<CAT$<HANZI>>` / …

频次 < `min_word_freq` 的训练词把它的 emission 计数同时加到自己的 SUF shape
列与 CAT shape 列上，这样 OOV 推理时无论查 SUF 还是 CAT 都有稳定支持。

### TnT 后缀树 (`--weight_tnt`)

Brants (2000) 的 OOV 处理：低频词 (freq ≤ `rare_threshold`) 按
后缀长度 k 递归平滑：

\[
P_{\text{smooth}}(t \mid s_k) = \frac{c(t, s_k) + \theta\,P_{\text{smooth}}(t\mid s_{k-1})}{c(s_k) + \theta}
\]

最长支持度的后缀给出 \( P(t\mid s) \)，按贝叶斯翻为相对发射
\( \log P(t\mid s) - \log P(t) \)。该项**只在 OOV 推理时**叠加在
shape 列的 log MLE 上：

\[
\log B(x\mid t) = \log P_{\text{shape}}(x\mid t) + w_{\text{tnt}} \cdot
\bigl[\log P(t\mid s) - \log P(t)\bigr]
\]

`weight_tnt = 0.3` 是经验值；英文按首字母大小写两支建独立的后缀树
（``upper`` / ``lower``）。

### 结构化约束 (`--no_constraints` 关闭)

自动检测标签方案后，给 `log_pi` / `log_trans` 加 -inf mask：

- BIO：禁止 `start → I-X`，禁止 `* → I-Y` 当前驱不是 `B-Y` / `I-Y`
- BMES：禁止 `start → M-/E-`；`B-X` / `M-X` 必须接 `M-X` / `E-X`；
  `M-X` / `E-X` 必须由 `B-X` / `M-X` 引出

### 设备选择

`hmm.resolve_backend("cpu" | "cuda")`：

- `cpu` → `numpy`，永不 import cupy
- `cuda` → `cupy`；CuPy / CUDA 不可用时**自动回退** CPU 并打印提示

`run.py` 在 import cupy 之前设置 `CUDA_VISIBLE_DEVICES`，所以
`--gpu 3` 不会抢占别人的卡。

---

## 实测结果（验证集 micro avg F1）

均在 `NER/Chinese/validation.txt` / `NER/English/validation.txt` 上评测，
等价 `python NER/check.py` 的 micro avg（已忽略 `O`）。

### 关键结果

| 配置 | Chinese F1 | English F1 |
| --- | --- | --- |
| 基线 1：order=1，无特征，α=0.01 | 0.8855 | 0.7873 |
| 基线 2：order=2 + 形态 OOV + 结构约束 | 0.8917 | 0.8246 |
| 基线 2 + `--tune`（上一版本默认） | 0.8926 | 0.8256 |
| **order=2 + TnT 后缀树 (默认)** | **0.8917** | **0.8494** |

**净增益（vs. 上一版本默认）**：Chinese 持平，**English +2.38 pp**。

### TnT 后缀树 ablation（English，order=2）

| weight_tnt | suffix_min_count | F1 (P / R / F1) |
| --- | --- | --- |
| 0    | 200 | 0.8470 / 0.8100 / 0.8280 |
| 0.3  | 100 | 0.8657 / 0.8177 / 0.8411 |
| 0.3  | 200 | 0.8693 / 0.8180 / **0.8429** |
| 0.5  | 200 | 0.8650 / 0.8201 / 0.8419 |
| 0.5  | 100 | 0.8645 / 0.8348 / **0.8494** |
| 1.0  | 200 | 0.8582 / 0.8133 / 0.8352 |

把 TnT 后缀树整个关掉得 0.8280；把它的混合权重设到经验最优 (0.5/100) 拿到
**0.8494**，相对**+2.14 pp**。

### 中文上 trigram 的 Brants λ

通常会出现：`λ1(uni)=0.005–0.05, λ2(bi)=0.30–0.45, λ3(tri)=0.55–0.70`，表示
二阶模型主要依赖 trigram 信号但保留 bigram 回退。

---

## 接口快速参考

```python
from hmm import HMM
from data_utils import read_conll, read_lines_with_blanks, group_into_sentences, write_predictions

train = read_conll("../NER/English/train.txt")
m = HMM(
    device="cpu",
    order=2,
    alpha_pi=1.0, alpha_trans=1e-2, alpha_emit=1e-4,
    use_suffix_oov=True, max_suffix_len=10, suffix_min_count=200,
    use_tnt_tree=True, weight_tnt=0.3,
    use_constraints=True,
)
m.fit(train)

records = read_lines_with_blanks("../NER/English/validation.txt")
sentences = group_into_sentences(records)
preds = m.predict(sentences, batch_size=64)
write_predictions("outputs/English_pred.txt", records, preds)
```
