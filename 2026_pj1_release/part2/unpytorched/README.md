# Part2 Bonus: unpytorched 纯手写 CNN 实现

本目录是纯手写轮子版本：
- 不使用 `torch`、`torchvision` 等深度学习框架
- 通过 `backend.py` 抽象数组库：**GPU 用 `cupy`，CPU 用 `numpy`**
- 手写参数系统、损失函数、优化器和训练循环
- 支持输出训练曲线和混淆矩阵
- 默认在 `GPU4` 上运行；无 CUDA 环境时可用 `--device cpu` 回退到纯 numpy

## 1. 安装依赖

```bash
pip install -r requirements.txt
```

- GPU 训练需要 `cupy`，如果你的 CUDA/驱动环境不同，可将 `cupy-cuda12x` 替换为对应版本的 cupy 包。
- CPU 训练只需要 `numpy`（`matplotlib` 已依赖），`cupy` 可以不装。

## 2. 目录结构

期望数据目录（与原工程一致）：

```text
part2/train/
  1/*.bmp
  2/*.bmp
  ...
  12/*.bmp
```

默认 `train.py` 读取 `../train`。

## 3. 代码结构

- `backend.py`：数组库抽象，根据 `UNPYTORCHED_DEVICE` 选择 `numpy` 或 `cupy`
- `mynn.py`：手写 `Parameter/Module`、`Conv2d/MaxPool2d/ReLU/Linear`、`CrossEntropyLoss`、`AdamW`
- `model.py`：基于手写层拼装 `HanziCNN`
- `train.py`：训练、验证、保存、绘图（CPU/GPU 两用）
- `predict.py`：推理（CPU/GPU 两用）

## 4. 网络结构（HanziCNN）

- 输入：`1 x 64 x 64`（灰度）
- 特征提取：
  - `Conv(1->16,3x3,pad=1) -> ReLU -> Conv(16->16,3x3,pad=1) -> ReLU -> MaxPool(2x2)`
  - `Conv(16->32,3x3,pad=1) -> ReLU -> Conv(32->32,3x3,pad=1) -> ReLU -> MaxPool(2x2)`
  - `Conv(32->64,3x3,pad=1) -> ReLU -> MaxPool(2x2)`
- 分类头：
  - `AdaptiveAvgPool2d(1,1) -> Flatten -> Linear(64,64) -> ReLU -> Dropout(0.2) -> Linear(64,12)`

所有层都在 `mynn.py` 手写 forward/backward：
- `Conv2d.backward`：显式计算 `dW`、`db`、`dx`
- `MaxPool2d.backward`：根据前向最大值位置回传梯度
- `CrossEntropyLoss`：手写 softmax + NLL 梯度
- `AdamW`：手写一阶/二阶动量更新

## 5. 训练

### 5.1 GPU 训练（默认）

```bash
python train.py --device gpu --gpu-id 4 --epochs 20 --batch-size 32
```

说明：
- `--device gpu` 为默认值，脚本会执行 `cp.cuda.Device(<gpu_id>).use()`。
- 依赖 CUDA 可用的 cupy 运行时。

### 5.2 CPU 训练

```bash
python train.py --device cpu --epochs 5 --batch-size 16
```

说明：
- CPU 模式下底层切换为 `numpy`，不需要安装 `cupy`。
- 由于 `Conv2d` 的前向/反向是显式 Python 循环（未用 im2col 或 cuDNN），
  CPU 速度**远慢于 GPU**，建议将 `--epochs` 和 `--batch-size` 调小用于调试/验证正确性，
  不建议用于完整训练。

## 6. 训练输出

默认保存在：
- `./checkpoints/best_model.npz`
- `./checkpoints/meta.json`
- `./checkpoints/history.json`
- `./checkpoints/class_names.json`

并保存图像到当前目录（可改 `--plot-dir`）：
- `loss_curve.png`
- `confusion_matrix.png`

## 7. 推理

GPU：

```bash
python predict.py --device gpu --gpu-id 4 --image ../train/1/609.bmp --checkpoint ./checkpoints/best_model.npz --meta ./checkpoints/meta.json
```

CPU：

```bash
python predict.py --device cpu --image ../train/1/609.bmp --checkpoint ./checkpoints/best_model.npz --meta ./checkpoints/meta.json
```

> 注：同一份 `best_model.npz` 在 GPU/CPU 两种后端间可以互通（`.npz` 本质是 numpy 格式）。
