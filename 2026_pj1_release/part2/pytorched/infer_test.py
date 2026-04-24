"""Batch inference on a full test set for the PyTorch CNN.

Usage
-----
Given a checkpoint ``best_model.pt`` and a test directory laid out as
``ImageFolder`` (``<test-dir>/<class_name>/*.bmp``), run forward on every
sample and report overall accuracy + per-class accuracy. A confusion matrix
PNG is also saved to ``--output-dir``.

Example
~~~~~~~
.. code-block:: bash

    python infer_test.py \
        --checkpoint ./checkpoints/best_model.pt \
        --test-dir ../test \
        --batch-size 128 \
        --device cuda

Class-name alignment
--------------------
The checkpoint stores the class order used at training time. If the test
directory's class folders match that order (name-by-name), the ckpt labels
are used directly. Otherwise the inference falls back to the test folder's
own class order and prints a warning, which typically hurts accuracy.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import HanziCNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch inference over a test folder and report accuracy + confusion matrix."
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pt")
    parser.add_argument(
        "--test-dir",
        type=str,
        required=True,
        help="Path to a test folder organised as ImageFolder (class_name/*.bmp).",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Inference device. Falls back to CPU if CUDA is unavailable.",
    )
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory to save confusion_matrix.png.",
    )
    parser.add_argument(
        "--class-names",
        type=str,
        default="",
        help="Optional class_names.json to override the ordering saved in the checkpoint.",
    )
    return parser.parse_args()


def select_device(device: str, gpu_id: int) -> torch.device:
    if device == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    if 0 <= gpu_id < torch.cuda.device_count():
        torch.cuda.set_device(gpu_id)
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cuda:0")


def load_model(ckpt_path: Path, device: torch.device) -> Tuple[torch.nn.Module, List[str], int]:
    ckpt = torch.load(ckpt_path, map_location=device)
    class_names = ckpt.get("class_names") or [str(i) for i in range(12)]
    img_size = int(ckpt.get("img_size", 64))
    model = HanziCNN(num_classes=len(class_names))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, list(class_names), img_size


def build_loader(test_dir: Path, img_size: int, batch_size: int, num_workers: int, pin: bool):
    tfm = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    dataset = datasets.ImageFolder(str(test_dir), transform=tfm)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=False,
    )
    return dataset, loader


def align_label_mapping(ckpt_classes: List[str], test_classes: List[str]) -> Tuple[List[int], List[str], bool]:
    """Return ``test_to_ckpt`` so that ``ckpt_label = test_to_ckpt[test_label]``.

    If every ckpt class is present in ``test_classes``, we can remap labels and
    accuracy remains meaningful. Otherwise we just identity-map and warn.
    """
    if ckpt_classes == test_classes:
        return list(range(len(test_classes))), ckpt_classes, True
    ckpt_idx = {c: i for i, c in enumerate(ckpt_classes)}
    if all(c in ckpt_idx for c in test_classes):
        mapping = [ckpt_idx[c] for c in test_classes]
        return mapping, ckpt_classes, True
    return list(range(len(test_classes))), test_classes, False


def run_inference(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    all_preds: List[np.ndarray] = []
    all_tgts: List[np.ndarray] = []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_tgts.append(targets.numpy())
    if not all_preds:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)
    return np.concatenate(all_preds), np.concatenate(all_tgts)


def build_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= int(t) < num_classes and 0 <= int(p) < num_classes:
            cm[int(t), int(p)] += 1
    return cm


def per_class_accuracy(cm: np.ndarray) -> np.ndarray:
    totals = cm.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        acc = np.where(totals > 0, np.diag(cm) / np.maximum(totals, 1), np.nan)
    return acc


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], accuracy: float, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True Label",
        xlabel="Predicted Label",
        title=f"Test Confusion Matrix  (acc={accuracy:.4f})",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    max_value = cm.max() if cm.size > 0 else 0
    threshold = max_value / 2.0 if max_value > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color=color, fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.checkpoint).resolve()
    test_dir = Path(args.test_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Test dir not found: {test_dir}")

    device = select_device(args.device, args.gpu_id)
    print(f"[device] {device}")

    model, ckpt_classes, img_size = load_model(ckpt_path, device)
    if args.class_names:
        with open(Path(args.class_names).resolve(), "r", encoding="utf-8") as f:
            ckpt_classes = list(json.load(f))

    pin = device.type == "cuda"
    dataset, loader = build_loader(test_dir, img_size, args.batch_size, args.num_workers, pin)
    test_classes = list(dataset.classes)

    mapping, report_classes, aligned = align_label_mapping(ckpt_classes, test_classes)
    if not aligned:
        print(
            "[warn] Test folder class names do not match / cover the ckpt's class names. "
            "Labels will be compared using the test-folder's own indexing, which likely "
            "yields nonsense accuracy. Make the test dir use the same class folder names "
            "as training to fix this.\n"
            f"       ckpt classes : {ckpt_classes}\n"
            f"       test classes : {test_classes}"
        )

    preds_raw, tgts_raw = run_inference(model, loader, device)
    if aligned:
        y_true = np.asarray([mapping[int(t)] for t in tgts_raw], dtype=np.int64)
    else:
        y_true = tgts_raw.astype(np.int64)
    y_pred = preds_raw.astype(np.int64)

    num_classes = len(report_classes)
    accuracy = float((y_pred == y_true).mean()) if y_true.size else 0.0
    cm = build_confusion_matrix(y_true, y_pred, num_classes)
    per_cls = per_class_accuracy(cm)

    print(f"Checkpoint : {ckpt_path}")
    print(f"Test dir   : {test_dir}")
    print(f"Num samples: {y_true.size}")
    print(f"Img size   : {img_size}x{img_size}")
    print(f"Overall accuracy: {accuracy:.4f}")
    print("Per-class accuracy:")
    for name, acc in zip(report_classes, per_cls):
        if np.isnan(acc):
            print(f"  {name:<12s}: (no samples)")
        else:
            print(f"  {name:<12s}: {acc:.4f}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cm_path = output_dir / f"test_confusion_matrix_{timestamp}.png"
    plot_confusion_matrix(cm, report_classes, accuracy, cm_path)
    print(f"Saved confusion matrix: {cm_path}")


if __name__ == "__main__":
    main()
