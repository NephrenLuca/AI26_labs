"""Batch inference on a full test set for the handwritten (unpytorched) CNN.

Usage
-----
Given a numpy-format checkpoint ``best_model.npz`` saved by ``train.py`` and a
test directory organised as ``<test-dir>/<class_name>/*.bmp``, compute the
forward pass over every image and report overall / per-class accuracy plus a
confusion-matrix PNG.

Example
~~~~~~~
.. code-block:: bash

    python infer_test.py \
        --device gpu --gpu-id 4 \
        --checkpoint ./checkpoints/best_model.npz \
        --meta ./checkpoints/meta.json \
        --test-dir ../test

CPU variant:

.. code-block:: bash

    python infer_test.py \
        --device cpu \
        --checkpoint ./checkpoints/best_model.npz \
        --meta ./checkpoints/meta.json \
        --test-dir ../test
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the handwritten CNN on a full test folder and report accuracy + confusion matrix."
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.npz")
    parser.add_argument("--meta", type=str, default="", help="Optional meta.json (class_names + img_size).")
    parser.add_argument(
        "--test-dir",
        type=str,
        required=True,
        help="Test folder with class subdirs: <test-dir>/<class_name>/*.bmp",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "gpu"],
        default="gpu",
        help="Backend: 'gpu' uses cupy, 'cpu' uses numpy.",
    )
    parser.add_argument("--gpu-id", type=int, default=4)
    parser.add_argument("--img-size", type=int, default=64, help="Fallback when --meta is not given.")
    parser.add_argument("--output-dir", type=str, default=".")
    parser.add_argument("--no-figure", action="store_true")
    return parser.parse_args()


# Backend must be fixed before importing ``backend`` / ``model`` / ``mynn``.
_ARGS = parse_args()
os.environ["UNPYTORCHED_DEVICE"] = _ARGS.device

from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from backend import xp as cp, is_gpu, set_gpu_device, to_numpy
from model import HanziCNN


def resize_nearest(img: cp.ndarray, size: int) -> cp.ndarray:
    h, w = img.shape
    ys = cp.linspace(0, h - 1, size).astype(cp.int32)
    xs = cp.linspace(0, w - 1, size).astype(cp.int32)
    return img[ys][:, xs]


def load_checkpoint(path: Path) -> dict:
    loaded = cp.load(path, allow_pickle=False)
    return {k: loaded[k] for k in loaded.files}


def read_meta(meta_path: str, img_size_fallback: int) -> Tuple[List[str], int]:
    if meta_path:
        with open(Path(meta_path).resolve(), "r", encoding="utf-8") as f:
            meta = json.load(f)
        class_names = list(meta.get("class_names", []))
        img_size = int(meta.get("img_size", img_size_fallback))
        if not class_names:
            raise RuntimeError("meta.json has empty class_names")
        return class_names, img_size
    return [], img_size_fallback


def discover_classes(test_dir: Path) -> List[str]:
    entries = [p.name for p in test_dir.iterdir() if p.is_dir()]
    try:
        return sorted(entries, key=lambda x: int(x))
    except ValueError:
        return sorted(entries)


def load_test_set(test_dir: Path, class_names: List[str], img_size: int) -> Tuple[cp.ndarray, cp.ndarray, List[Path]]:
    valid_ext = {".bmp", ".png", ".jpg", ".jpeg"}
    images: List[cp.ndarray] = []
    labels: List[int] = []
    paths: List[Path] = []
    cls_to_idx = {c: i for i, c in enumerate(class_names)}

    for cls in class_names:
        cls_dir = test_dir / cls
        if not cls_dir.is_dir():
            continue
        for img_path in sorted(cls_dir.iterdir()):
            if img_path.suffix.lower() not in valid_ext:
                continue
            img = plt.imread(img_path)
            img = cp.asarray(img)
            if img.ndim == 3:
                img = img.mean(axis=2)
            img = img.astype(cp.float32)
            if img.max() > 1.0:
                img = img / 255.0
            img = resize_nearest(img, img_size)
            img = (img - 0.5) / 0.5
            images.append(img[None, :, :])
            labels.append(cls_to_idx[cls])
            paths.append(img_path)

    if not images:
        raise RuntimeError(f"No images found under {test_dir} for classes {class_names}")

    x = cp.stack(images).astype(cp.float32)
    y = cp.asarray(labels, dtype=cp.int64)
    return x, y, paths


def batched_forward(model: HanziCNN, x: cp.ndarray, batch_size: int) -> cp.ndarray:
    n = x.shape[0]
    parts: List[cp.ndarray] = []
    for start in range(0, n, batch_size):
        xb = x[start : start + batch_size]
        logits = model.forward(xb)
        parts.append(cp.argmax(logits, axis=1))
    return cp.concatenate(parts, axis=0) if parts else cp.zeros((0,), dtype=cp.int64)


def build_confusion_matrix_np(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
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
    args = _ARGS
    if is_gpu:
        set_gpu_device(args.gpu_id)
        print(f"[device] Using cupy on GPU{args.gpu_id}")
    else:
        print("[device] Using numpy on CPU")

    ckpt_path = Path(args.checkpoint).resolve()
    test_dir = Path(args.test_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Test dir not found: {test_dir}")

    class_names, img_size = read_meta(args.meta, args.img_size)
    if not class_names:
        class_names = discover_classes(test_dir)
        print(f"[meta] Using class names inferred from test dir: {class_names}")
    else:
        print(f"[meta] Using class names from meta.json: {class_names}")

    model = HanziCNN(num_classes=len(class_names))
    state = load_checkpoint(ckpt_path)
    model.load_state_dict(state)
    model.eval()

    x, y, _paths = load_test_set(test_dir, class_names, img_size)
    print(f"Loaded {x.shape[0]} test images, img_size={img_size}")

    preds = batched_forward(model, x, args.batch_size)

    y_true = to_numpy(y).astype(np.int64)
    y_pred = to_numpy(preds).astype(np.int64)

    accuracy = float((y_pred == y_true).mean()) if y_true.size else 0.0
    cm = build_confusion_matrix_np(y_true, y_pred, len(class_names))
    per_cls = per_class_accuracy(cm)

    print(f"Checkpoint : {ckpt_path}")
    print(f"Test dir   : {test_dir}")
    print(f"Num samples: {y_true.size}")
    print(f"Overall accuracy: {accuracy:.4f}")
    print("Per-class accuracy:")
    for name, acc in zip(class_names, per_cls):
        if np.isnan(acc):
            print(f"  {name:<12s}: (no samples)")
        else:
            print(f"  {name:<12s}: {acc:.4f}")

    if not args.no_figure:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        cm_path = output_dir / f"test_confusion_matrix_{ts}.png"
        plot_confusion_matrix(cm, class_names, accuracy, cm_path)
        print(f"Saved confusion matrix: {cm_path}")


if __name__ == "__main__":
    main()
