# 2026 课程项目 Part 1：手写反向传播 MLP

本仓库实现**多层感知机（MLP）**及**手动反向传播**，用于课程第一部分的两类任务：**回归**（拟合 \(y=\sin(x)\)，\(x\in[-\pi,\pi]\)）与**分类**（12 类手写汉字）。按课程要求，**不得使用 PyTorch / TensorFlow / JAX 等带自动微分的深度学习框架**；数值计算使用 **NumPy**（CPU），可选 **CuPy**（GPU，仅 ndarray 与算子，不提供 autograd）。

更完整的评分与提交说明见根目录 [`PJ1.md`](PJ1.md)。

---

## 神经网络结构

### 整体：可组合的 MLP

- **类**：`part1.network.MLP`
- **拓扑**：按 `layer_sizes` 依次堆叠 **Linear → 激活**（隐藏层），最后一层仅为 **Linear**（无激活），输出：
  - **回归**：标量或向量预测（如 `sin` 任务为 1 维）
  - **分类**：各类 **logits**（未经 softmax），与 `CrossEntropyLoss` 配合

### 基本单元（`part1.layers`）

| 模块 | 作用 |
|------|------|
| `Linear` | 仿射变换 \(y = xW + b\)；权重为 **Glorot uniform** 初始化；隐藏层 bias 可设为略负（课程建议） |
| `Tanh` | 双曲正切，逐元素；适合平滑目标（如 sin 回归） |
| `ReLU` | \(\max(0,x)\)，逐元素；分类任务常用 |
| `Sigmoid` | 可选，用于有界输出等场景 |

### 前向与反向

- **前向**：`MLP.forward(x)` 按层顺序调用各模块 `forward`。
- **反向**：损失对输出的梯度经 `MLP.backward(grad_out)` **从后往前**逐层 `backward`，与链式法则一致；`Linear` 在 `backward` 中计算并保存 `dW`、`db`。

### 损失函数（`part1.losses`）

- **`MSELoss`**：均方误差，用于回归；`forward(pred, target)` 后 `backward()` 得到 \(\partial L/\partial \text{pred}\)。
- **`CrossEntropyLoss`**：多类 **softmax + 交叉熵**；输入为 logits；支持整型标签 `(N,)` 或 one-hot `(N, C)`；`backward()` 返回对 logits 的梯度。

### 优化与学习率（`part1.optimizer`）

- **`SGDOptimizer`**：带动量 SGD，可选 **L2 权重衰减**（仅作用在权重 \(W\) 上）。
- **`make_lr_fn`**：按 epoch 的学习率策略 — `constant` / `step` / `cosine`。

### 计算后端（`part1.backend`）

- **`init_backend(use_cuda=..., gpu_id=...)`**：全局选择 **NumPy** 或 **CuPy**；后续张量与算子均通过 `get_xp()` 取得与 NumPy 兼容的 API。
- 训练逻辑只写一套；GPU 需单独安装与 CUDA 版本匹配的 CuPy wheel（见下文）。

### 默认超参参考（`part1.hyperparams`）

- **`SinRegressionHParams`**：如 `layer_sizes=(1, 128, 128, 1)`，`hidden_activation="tanh"`，余弦学习率等。
- **`CharClassificationHParams`**：如隐藏层 `(512, 256, 128)`，`relu`，步长衰减、动量、weight decay 等。

---

## 项目目录结构

```text
2026_pj1_release/
├── PJ1.md                 # 课程说明：任务、评分、提交要求
├── README.md              # 本文件
└── part1/                 # Part 1 代码包（手写 BP + MLP）
    ├── __init__.py
    ├── requirements.txt
    ├── backend.py         # NumPy / CuPy 后端与设备
    ├── layers.py          # Linear, Tanh, ReLU, Sigmoid
    ├── network.py         # MLP 组装与前向/反向
    ├── losses.py          # MSE、Softmax 交叉熵
    ├── optimizer.py       # SGD + 动量、学习率调度
    ├── hyperparams.py     # 回归/分类默认超参 dataclass
    ├── utils.py           # 随机种子、梯度 L2 裁剪
    ├── data_sin.py        # sin 任务：评估网格 MAE 等
    ├── data_chars.py      # 汉字数据：目录加载 / 合成测试数据
    ├── train_regression.py   # 回归任务训练入口（CLI）
    └── train_classification.py  # 分类任务训练入口（CLI）
```

---

## 各 Python 文件说明

| 文件 | 用途 |
|------|------|
| **`part1/__init__.py`** | 包说明；导出子模块名（`backend`, `layers`, `network`, `losses`, `optimizer`）。 |
| **`part1/backend.py`** | 初始化全局数组库（NumPy 或 CuPy）、是否 CUDA、`get_xp()`、`to_cpu_array()`、可选 `device_scope`。 |
| **`part1/layers.py`** | 可微分层：`forward` / `backward`；Linear 的 Xavier 初始化与梯度 `dW`、`db`。 |
| **`part1/network.py`** | `MLP`：根据 `layer_sizes` 与 `hidden_activation` 串联 Linear 与激活；`linear_layers()` 供优化器收集参数。 |
| **`part1/losses.py`** | `MSELoss`、`CrossEntropyLoss`（数值稳定 softmax）；与网络最后一层 logits 衔接。 |
| **`part1/optimizer.py`** | `SGDOptimizer`；`lr_step`、`lr_cosine`、`make_lr_fn`。 |
| **`part1/hyperparams.py`** | `SinRegressionHParams`、`CharClassificationHParams` 及默认实例 `SIN_DEFAULTS`、`CHAR_DEFAULTS`。 |
| **`part1/utils.py`** | `set_random_seed`（含 CuPy）、`clip_tensor_l2_norm`（反传前裁剪梯度）。 |
| **`part1/data_sin.py`** | `eval_grid_mae`：在 \([-\pi,\pi]\) 上随机点估计 MAE（对照课程「平均误差 < 0.01」）。 |
| **`part1/data_chars.py`** | `load_char_dataset`：按子文件夹读图、resize、灰度、展平；`make_synthetic_char_dataset`：随机图冒烟测试。 |
| **`part1/train_regression.py`** | 回归任务 CLI：\([-\pi,\pi]\) 上拟合 \(\sin(x)\)，默认超参见 `hyperparams.SIN_DEFAULTS`。 |
| **`part1/train_classification.py`** | 分类任务 CLI：12 类汉字 MLP；`--data` 读目录或 `--synthetic` 冒烟测试。 |

---

## 使用方法

### 1. 环境

- Python 3.10+ 推荐。
- 在项目根目录或 `part1` 上级安装依赖：

```bash
pip install -r part1/requirements.txt
```

- **可选 GPU**：安装与本地 CUDA 匹配的 CuPy，例如 CUDA 12.x：

```bash
pip install cupy-cuda12x
```

无 CuPy 时仅使用 CPU（NumPy）。

### 2. 数据

- **回归**：无需文件；训练时可在 \([-\pi,\pi]\) 上随机采样 \(x\)，目标 \(\sin(x)\)。
- **分类**：准备目录结构（类名为子文件夹名，字典序对应标签 0…K-1）：

```text
data_root/
  0/   *.png ...
  1/
  ...
  11/
```

使用 `load_char_dataset("data_root", image_size=(28, 28), num_classes=12)` 得到 `X, y, class_names`。训练入口脚本会在内部把数据搬到当前 backend（NumPy / CuPy）。

### 3. 训练入口（推荐）

在**项目根目录** `2026_pj1_release/` 下执行，使 `part1` 可作为包被解析：

```bash
cd path/to/2026_pj1_release
```

**GPU**：需已安装与 CUDA 版本匹配的 CuPy；用 `--gpu N` 指定**物理设备号**（如单机多卡上的 `4`、`5`、`6`、`7`）。仅用 CPU 时加 `--no-cuda`。

---

#### 3.1 回归：[`part1/train_regression.py`](part1/train_regression.py)

任务：在 \([-\pi,\pi]\) 上采样 \(x\)，用 MSE 拟合 \(y=\sin(x)\)。训练过程中按 `--log-every` 打印 **eval_mae**（随机点上的平均绝对误差），可与课程要求「平均误差 < 0.01」对照。

```bash
# 默认超参（见 hyperparams.SinRegressionHParams），CPU
python -m part1.train_regression --no-cuda

# 指定 GPU（示例：物理卡 5）
python -m part1.train_regression --gpu 5

# 常见调参：网络宽度、训练轮数、学习率与调度
python -m part1.train_regression --gpu 6 --epochs 6000 --layers 1,256,256,1 --lr 0.02 --lr-schedule cosine

# 完整参数说明
python -m part1.train_regression --help
```

| 参数（节选） | 说明 |
|--------------|------|
| `--no-cuda` | 强制 NumPy CPU，忽略 `--gpu`。 |
| `--gpu ID` | CuPy 使用的 CUDA 设备序号。 |
| `--layers` | 逗号分隔各层宽度，如 `1,128,128,1`（输入/输出一般为 1）。 |
| `--activation` | 隐藏层激活：`tanh`（默认）或 `relu`。 |
| `--epochs` / `--batch-size` / `--samples-per-epoch` | 训练规模与每 epoch 随机样本量。 |
| `--lr` / `--lr-schedule` | 学习率与 `constant` \| `step` \| `cosine` 调度。 |
| `--grad-clip` | 对损失关于输出的梯度做 L2 裁剪；设为负数可关闭。 |
| `--mae-threshold` | 日志中与 eval_mae 对比的阈值（默认 0.01）。 |
| `--exit-on-mae-fail` | 若最终 MAE ≥ 阈值则以退出码 1 结束（便于 CI / 批处理）。 |

---

#### 3.2 分类：[`part1/train_classification.py`](part1/train_classification.py)

任务：12 类手写汉字，flatten 后接 MLP + softmax 交叉熵（实现见 `CrossEntropyLoss`）。数据为「根目录下每类一个子文件夹」，**子文件夹名按字典序**对应标签 `0 … K-1`（见上文目录示例）。

```bash
# 真实数据：将 --data 指向含 12 个子文件夹的根目录
python -m part1.train_classification --data path/to/train_root

# 同上，使用 GPU 7
python -m part1.train_classification --data path/to/train_root --gpu 7

# 不读盘：随机图像冒烟测试（不反映真实精度）
python -m part1.train_classification --synthetic --epochs 5 --no-cuda

# 划分验证集并打印 val_acc
python -m part1.train_classification --data path/to/train_root --val-ratio 0.1 --gpu 4

# 完整参数说明
python -m part1.train_classification --help
```

| 参数（节选） | 说明 |
|--------------|------|
| `--data DIR` | 分类数据根目录（与 `--synthetic` 二选一）。 |
| `--synthetic` | 使用合成随机图像与平衡标签，用于打通训练流程。 |
| `--img-size H,W` | resize 尺寸，默认 `28,28`。 |
| `--hidden` | 隐藏层宽度，如 `512,256,128`。 |
| `--num-classes` | 类别数，默认 12。 |
| `--val-ratio` | 验证集比例；`0` 表示全量训练，仅报告 train_acc。 |
| `--gpu` / `--no-cuda` | 同回归脚本。 |

---

### 4. 自行编写训练循环（可选）

若需自定义数据管线或记录方式，可直接调用 `backend.init_backend`、`MLP`、`MSELoss` / `CrossEntropyLoss`、`SGDOptimizer`、`make_lr_fn`、`utils.set_random_seed` 与 `clip_tensor_l2_norm`，模式与入口脚本内部一致。若从非根目录运行且无法 `import part1`，可将项目根加入 `PYTHONPATH`：

```bash
# Windows CMD
set PYTHONPATH=c:\Users\14144\Desktop\AIpj\2026_pj1_release

# Linux / macOS
export PYTHONPATH=/path/to/2026_pj1_release
```
