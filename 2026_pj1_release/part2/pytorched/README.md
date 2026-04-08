# Part2 PyTorch CNN - Handwritten Hanzi Classification

This directory contains a complete PyTorch implementation for the Part2 task:
- 12-class handwritten Hanzi classification
- custom CNN (no pretrained model)
- single-GPU and multi-GPU training

## 1. Environment

```bash
pip install -r requirements.txt
```

## 2. Data layout

Expected ImageFolder structure:

```text
part2/train/
  1/*.bmp
  2/*.bmp
  ...
  12/*.bmp
```

By default, `train.py` reads `../train` (relative to `pytorched`), which matches your current project layout.

## 3. Train

### Single GPU example

```bash
python train.py --device cuda --gpu-ids 0 --epochs 40 --batch-size 128
```

### Use GPU 4-7 (your scenario)

```bash
python train.py --device cuda --gpu-ids 4,5,6,7 --epochs 50 --batch-size 256
```

If your runtime uses `CUDA_VISIBLE_DEVICES`, keep `--gpu-ids` consistent with visible indices.

## 4. Outputs

Training saves files into `./checkpoints`:
- `best_model.pt`: best validation checkpoint
- `history.json`: per-epoch metrics
- `class_names.json`: label-name mapping

Also saves matplotlib figures into current working directory:
- `loss_curve.png`: train/val loss vs epoch
- `confusion_matrix.png`: final validation confusion matrix (from best checkpoint)

## 5. Inference

```bash
python predict.py --image ../train/1/609.bmp --checkpoint ./checkpoints/best_model.pt --device cuda
```

`predict.py` reads images via `matplotlib` (no direct `PIL` import in project code).

## 6. Important implementation details

- Model: custom `HanziCNN` in `models.py` (Conv-BN-ReLU blocks + pooling + classifier).
- Data split: stratified train/val split from the training folder.
- Loss: cross entropy.
- Optimizer: AdamW.
- LR schedule: cosine annealing.
- Overfitting mitigation (non-bonus): augmentation, weight decay, dropout, batch norm.
