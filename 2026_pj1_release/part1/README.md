# Part1: 手写反向传播神经网络

本目录实现了一个不依赖深度学习框架（如 PyTorch）的多层全连接神经网络，用于完成：

- 回归任务：拟合 `y = sin(x)`，`x in [-pi, pi]`
- 分类任务：12 类手写汉字分类（`train` 目录下按类别分文件夹）

实现包含手动前向传播与反向传播，并支持 `BatchNorm`、`Dropout`、`数据增强`、`学习率调度`、`最佳模型保存`、`CPU/GPU`（NumPy/CuPy）。

---

## 目录说明

- `nn.py`：核心 `NeuralNetwork` 实现（Linear + 激活 + BN + Dropout + 反向传播）
- `train_regression.py`：回归训练脚本
- `train_classification.py`：分类训练脚本
- `infer_classification.py`：加载已有 ckpt 进行推理 / 验证集评估 / 单图预测
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
- **训练集数据增强**（仅分类）：随机旋转/平移/缩放/轻噪声（仅作用于 train，不污染 val）
- **学习率调度**（仅分类）：
  - `plateau`：验证集指标停滞时自动降学习率
  - `cosine`：余弦退火，随 epoch 平滑衰减学习率
- **最佳模型保存与恢复**（仅分类）：按验证集准确率保存 best checkpoint，训练结束默认恢复 best 参数后再评估

### 3) GPU 支持（本地 GPU4-7）

- 通过 `--gpu 4/5/6/7` 指定设备号
- 通过 `--no-cuda` 强制使用 CPU
- 若缺少 CuPy，将提示安装

### 4) 训练结果可视化

- 回归任务：训练结束后在当前目录保存 `regression_loss_*.png`（loss-epoch 曲线）
- 分类任务：训练结束后在当前目录保存 `classification_confusion_*.png`（混淆矩阵 + 验证集准确率）
- 分类任务同时保存 `classification_best_*.npz`（best checkpoint）

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
python -m part1.train_regression --gpu 4 --epochs 1000 --batch-size 128 --lr 0.01 --activation tanh
```

常用参数：

- `--hidden 64,64`
- `--batchnorm`
- `--dropout 0.1`
- `--gpu 4`（可改成 `5/6/7`）
- `--no-cuda`（改为 CPU）

### 分类训练

```bash
python -m part1.train_classification --data-dir part1/train --gpu 4 --epochs 200 --batch-size 128 --lr 0.003 --hidden 1024,512,256 --batchnorm --dropout 0.15 --scheduler plateau --lr-min 1e-5 --plateau-factor 0.5 --plateau-patience 10 --plateau-min-delta 1e-4 --aug-prob 0.9 --aug-rotate 10 --aug-translate 0.06 --aug-scale-min 0.95 --aug-scale-max 1.08 --aug-noise-std 0.02
```

常用参数：

- `--img-size 28,28`
- `--val-ratio 0.2`
- `--batchnorm`
- `--dropout 0.2`
- `--gpu 4`（可改成 `5/6/7`）
- `--no-cuda`
- `--no-augment`：关闭训练集数据增强
- `--aug-prob 0.8`：每张图触发增强的概率
- `--aug-rotate 12`：随机旋转最大角度（度）
- `--aug-translate 0.08`：随机平移比例（相对宽高）
- `--aug-scale-min 0.92 --aug-scale-max 1.10`：随机缩放范围
- `--aug-noise-std 0.03`：高斯噪声标准差
- `--scheduler none/plateau/cosine`：学习率调度策略
- `--lr-min 1e-5`：最小学习率
- `--plateau-factor 0.5`：plateau 触发后学习率衰减倍率
- `--plateau-patience 12`：验证集无提升多少轮后降学习率
- `--plateau-min-delta 1e-4`：最小提升阈值（小于该值视为未提升）
- `--no-restore-best`：训练结束不恢复 best 参数（默认会恢复）

推荐的调度策略：

- `plateau`：默认推荐，验证集停滞时自动降学习率，适合冲高准确率
- `cosine`：学习率平滑下降，训练过程更稳定

查看完整参数：

```bash
python -m part1.train_regression --help
python -m part1.train_classification --help
```

### 使用已有 ckpt 推理 / 验证

训练会保存 `classification_best_*.npz`。使用 `infer_classification.py` 可以在不重新训练的情况下加载该 ckpt 并进行评估或单图预测。

在整个数据集上评估：

```bash
python -m part1.infer_classification --ckpt part1/classification_best_20260413_134341.npz --data-dir part1/train --no-cuda
```

复现训练时的验证集并评估（需保持 `--val-ratio` 与 `--seed` 与训练时相同）：

```bash
python -m part1.infer_classification --ckpt part1/classification_best_20260413_134341.npz --data-dir part1/train --split val --val-ratio 0.2 --seed 42 --no-cuda
```

预测单张图片（打印 top-k 概率）：

```bash
python -m part1.infer_classification --ckpt part1/classification_best_20260413_134341.npz --data-dir part1/train --image path/to/some.bmp --topk 3 --no-cuda
```

常用参数：

- `--ckpt`：必填，`classification_best_*.npz` 路径
- `--data-dir`：仅用于读取类别名（子文件夹顺序即 label 顺序），默认评估该目录
- `--image`：若提供，则只对单张图做预测，不做整体评估
- `--split all/val/train`：评估子集；`val`/`train` 需与训练一致的 `--val-ratio` `--seed`
- `--img-size 28,28`：必须与训练时一致（ckpt 会自动校验 `H*W` 与输入维度）
- `--activation relu/tanh/sigmoid`：必须与训练时一致（ckpt 不记录激活函数）
- `--batchnorm auto/on/off`：默认 `auto`，从 ckpt 的 running 统计量自动判断
- `--save-misclassified`：在 `--split all` 模式下，将误分类样本写入 CSV
- `--no-figure`：不保存混淆矩阵图片
- `--gpu 4` / `--no-cuda`：设备选择
- `--batch-size 512`：推理批大小

输出：

- 终端打印整体准确率、每类准确率、网络结构、BN 是否启用
- 默认保存 `classification_infer_confusion_*.png`（混淆矩阵）
- 可选保存 `classification_infer_misclassified_*.csv`（误分类清单）

---

## 输出结果

训练成功后终端会打印：

- 最终训练 loss（回归）或最终验证集准确率（分类）
- 分类每个 epoch 的 `lr / val_loss / val_acc / best` 日志
- 分类最佳验证准确率及对应 epoch
- 分类 best checkpoint 路径（`classification_best_*.npz`）
- 图像保存路径（保存在 `part1` 目录下）

