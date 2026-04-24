# 第二部分 PyTorch CNN - 手写汉字分类

本目录提供了第二部分任务的完整 PyTorch 实现，主要特性如下：
- 12 类手写汉字分类
- 自定义 CNN（不使用预训练模型）
- 支持单卡与多卡 GPU 训练

## 1. 环境安装

```bash
pip install -r requirements.txt
```

## 2. 数据目录结构

程序使用 `ImageFolder` 格式读取数据，目录应为：

```text
part2/train/
  1/*.bmp
  2/*.bmp
  ...
  12/*.bmp
```

默认情况下，`train.py` 会读取 `../train`（相对于 `pytorched` 目录），与你当前工程结构一致。

## 3. 训练

### 单卡训练示例

```bash
python train.py --device cuda --gpu-ids 0 --epochs 40 --batch-size 128
```

### 使用 GPU 4-7 训练

```bash
python train.py --device cuda --gpu-ids 4,5,6,7 --epochs 50 --batch-size 256
```

如果运行环境设置了 `CUDA_VISIBLE_DEVICES`，请保证 `--gpu-ids` 与可见卡编号一致。

## 4. 模型结构说明（HanziCNN）

模型定义在 `models.py`，整体结构为“4 个卷积块 + 全局池化 + 两层全连接分类头”。

### 4.1 输入
- 输入通道：1（灰度图）
- 默认输入尺寸：`64 x 64`（由 `train.py` 的 `--img-size` 控制）

### 4.2 特征提取部分（4 个 ConvBlock）

每个 `ConvBlock` 结构：
- `Conv2d(3x3, padding=1, bias=False)`
- `BatchNorm2d`
- `ReLU`
- `Conv2d(3x3, padding=1, bias=False)`
- `BatchNorm2d`
- `ReLU`
- `MaxPool2d(2x2)`
- 可选 `Dropout2d`

4 个块的通道变化与 Dropout：
- Block1：`1 -> 32`，`Dropout2d(p=0.05)`
- Block2：`32 -> 64`，`Dropout2d(p=0.10)`
- Block3：`64 -> 128`，`Dropout2d(p=0.15)`
- Block4：`128 -> 256`，`Dropout2d(p=0.20)`

若输入为 `64 x 64`，经过 4 次 `MaxPool2d(2x2)` 后，特征图尺寸变为 `4 x 4`。

### 4.3 分类头
- `AdaptiveAvgPool2d((1,1))`：将 `256 x 4 x 4` 压缩为 `256 x 1 x 1`
- `Flatten`
- `Linear(256, 128) + ReLU + Dropout(0.3)`
- `Linear(128, num_classes)`，其中 `num_classes=12`

最终输出为 12 维 logits，用于交叉熵损失训练。

## 5. 训练输出

训练过程会在 `./checkpoints` 下保存：
- `best_model.pt`：验证集最优模型
- `history.json`：每个 epoch 的指标记录
- `class_names.json`：类别索引与类别名映射

并在当前运行目录保存 matplotlib 图像：
- `loss_curve.png`：训练/验证 loss 随 epoch 变化曲线
- `confusion_matrix.png`：基于最优模型计算的最终验证集混淆矩阵

## 6. 推理

### 6.1 单张图片推理

```bash
python predict.py --image ../train/1/609.bmp --checkpoint ./checkpoints/best_model.pt --device cuda
```

`predict.py` 使用 `matplotlib` 读取图片，不直接依赖 `Pillow`。

### 6.2 对整个测试集批量推理

给定 ckpt 与按 `ImageFolder` 结构组织的测试目录（`<test-dir>/<class_name>/*.bmp`），
`infer_test.py` 会对整个测试集做前向，打印整体准确率与每类准确率，并保存混淆矩阵 PNG：

```bash
python infer_test.py \
    --checkpoint ./checkpoints/best_model.pt \
    --test-dir ../test \
    --batch-size 128 \
    --device cuda
```

说明：

- ckpt 里保存了训练时的类别顺序。若测试目录的类别文件夹名与训练目录一致（例如都是 `1..12`），脚本会自动对齐标签。
- 测试目录若缺少某些训练类别，也可以正常评估，只是「缺席」类别将无样本。
- 输出 `test_confusion_matrix_<timestamp>.png`（保存到 `--output-dir`，默认当前目录）。

## 7. 训练细节

- 数据划分：分层 train/val 划分
- 损失函数：`CrossEntropyLoss`
- 优化器：`AdamW`
- 学习率策略：`CosineAnnealingLR`
- 抑制过拟合（非 Bonus）：数据增强、权重衰减、Dropout、BatchNorm
