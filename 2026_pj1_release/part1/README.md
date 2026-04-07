# Part1: 手写反向传播神经网络

本目录实现了一个不依赖深度学习框架（如 PyTorch）的多层全连接神经网络，用于完成：

- 回归任务：拟合 `y = sin(x)`，`x in [-pi, pi]`
- 分类任务：12 类手写汉字分类（`train` 目录下按类别分文件夹）

实现包含手动前向传播与反向传播，并支持 `BatchNorm`、`Dropout`、`CPU/GPU`（NumPy/CuPy）。

---

## 目录说明

- `nn.py`：核心 `NeuralNetwork` 实现（Linear + 激活 + BN + Dropout + 反向传播）
- `train_regression.py`：回归训练脚本
- `train_classification.py`：分类训练脚本
- `requirements.txt`：基础依赖
- `__init__.py`：包导出

---

## 已实现功能

### 1) 可配置网络结构

- 支持任意层数全连接网络（通过 `layer_sizes` 指定）
- 隐藏层激活支持：`relu` / `tanh` / `sigmoid`
- 输出层激活支持：`linear` / `tanh` / `sigmoid`

### 2) 训练稳定性与泛化

- **BatchNorm**：可在隐藏层启用（训练使用 batch 统计量，推理使用 running 统计量）
- **Dropout**：训练时启用，推理时自动关闭

### 3) GPU 支持（本地 GPU4-7）

- 通过 `--gpu 4/5/6/7` 指定设备号
- 通过 `--no-cuda` 强制使用 CPU
- 若缺少 CuPy，将提示安装

### 4) 训练结果可视化

- 回归任务：训练结束后在当前目录保存 `regression_loss_*.png`（loss-epoch 曲线）
- 分类任务：训练结束后在当前目录保存 `classification_confusion_*.png`（混淆矩阵 + 验证集准确率）

---

## 数据准备（分类）

默认分类脚本读取 `part1/train`，目录结构示例：

```text
part1/train/
  class0/
    *.bmp
  class1/
    *.bmp
  ...
  class11/
    *.bmp
```

说明：

- 每个子文件夹对应一个类别
- 脚本会按文件夹名字典序分配标签 `0..K-1`
- 支持图片后缀：`.bmp` `.png` `.jpg` `.jpeg`

---

## 环境安装

在 `2026_pj1_release` 根目录执行：

```bash
pip install -r part1/requirements.txt
```

若使用 GPU，请按本机 CUDA 版本安装对应 CuPy（示例）：

```bash
pip install cupy-cuda12x
```

---

## 使用方法

建议在 `2026_pj1_release` 根目录运行（保证 `python -m part1.xxx` 可解析）。

### 回归训练

```bash
python -m part1.train_regression --gpu 4 --epochs 2000 --batch-size 128 --lr 0.01 --activation tanh --batchnorm --dropout 0.1
```

常用参数：

- `--hidden 64,64`
- `--batchnorm`
- `--dropout 0.1`
- `--gpu 4`（可改成 `5/6/7`）
- `--no-cuda`（改为 CPU）

### 分类训练

```bash
python -m part1.train_classification --data-dir part1/train --gpu 7 --epochs 60 --batch-size 64 --lr 0.01 --hidden 256,128 --batchnorm --dropout 0.3
```

常用参数：

- `--img-size 28,28`
- `--val-ratio 0.2`
- `--batchnorm`
- `--dropout 0.2`
- `--gpu 4`（可改成 `5/6/7`）
- `--no-cuda`

查看完整参数：

```bash
python -m part1.train_regression --help
python -m part1.train_classification --help
```

---

## 输出结果

训练成功后终端会打印：

- 最终训练 loss（回归）或最终验证集准确率（分类）
- 图像保存路径（保存在 `part1` 目录下）

