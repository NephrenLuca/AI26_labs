import argparse
import os


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pure handwritten CNN training with numpy/cupy.")
    parser.add_argument("--data-dir", type=str, default="../train")
    parser.add_argument("--save-dir", type=str, default="./checkpoints")
    parser.add_argument("--plot-dir", type=str, default=".")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "gpu"],
        default="gpu",
        help="Backend: 'gpu' uses cupy, 'cpu' uses numpy.",
    )
    parser.add_argument("--gpu-id", type=int, default=4, help="Physical GPU id (only used when --device gpu).")
    return parser.parse_args()


# The backend (numpy vs cupy) must be decided **before** importing modules
# that rely on it (mynn / model). We therefore parse CLI args first and set
# UNPYTORCHED_DEVICE, then import the rest of the stack.
_ARGS = parse_args()
os.environ["UNPYTORCHED_DEVICE"] = _ARGS.device

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from backend import xp as cp, is_gpu, set_gpu_device, to_numpy, seed as backend_seed
from model import HanziCNN
from mynn import AdamW, CrossEntropyLoss


def set_seed(value: int) -> None:
    random.seed(value)
    backend_seed(value)


def resize_nearest(img: cp.ndarray, size: int) -> cp.ndarray:
    h, w = img.shape
    ys = cp.linspace(0, h - 1, size).astype(cp.int32)
    xs = cp.linspace(0, w - 1, size).astype(cp.int32)
    return img[ys][:, xs]


def load_dataset(data_dir: Path, img_size: int) -> Tuple[cp.ndarray, cp.ndarray, List[str]]:
    class_names = sorted([p.name for p in data_dir.iterdir() if p.is_dir()], key=lambda x: int(x))
    images: List[cp.ndarray] = []
    labels: List[int] = []
    for cls_idx, cls_name in enumerate(class_names):
        cls_dir = data_dir / cls_name
        for img_path in cls_dir.iterdir():
            if img_path.suffix.lower() != ".bmp":
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
            labels.append(cls_idx)

    x = cp.stack(images).astype(cp.float32)
    y = cp.asarray(labels, dtype=cp.int64)
    return x, y, class_names


def stratified_split(y: cp.ndarray, num_classes: int, val_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    rng = random.Random(seed)
    train_idx: List[int] = []
    val_idx: List[int] = []
    for cls in range(num_classes):
        idxs = cp.where(y == cls)[0].tolist()
        rng.shuffle(idxs)
        split = max(1, int(len(idxs) * val_ratio))
        val_idx.extend(idxs[:split])
        train_idx.extend(idxs[split:])
    return train_idx, val_idx


def iter_batches(indices: List[int], batch_size: int, shuffle: bool = True):
    idx = indices[:]
    if shuffle:
        random.shuffle(idx)
    for i in range(0, len(idx), batch_size):
        yield idx[i : i + batch_size]


def evaluate(model: HanziCNN, criterion: CrossEntropyLoss, x: cp.ndarray, y: cp.ndarray, indices: List[int], batch_size: int):
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    preds_all: List[cp.ndarray] = []
    tgts_all: List[cp.ndarray] = []
    for b in iter_batches(indices, batch_size, shuffle=False):
        xb = x[b]
        yb = y[b]
        logits = model.forward(xb)
        loss = criterion.forward(logits, yb)
        preds = cp.argmax(logits, axis=1)
        loss_sum += loss * len(b)
        correct += int((preds == yb).sum().item())
        total += len(b)
        preds_all.append(preds)
        tgts_all.append(yb)
    return loss_sum / max(total, 1), correct / max(total, 1), cp.concatenate(tgts_all), cp.concatenate(preds_all)


def build_confusion_matrix(targets: cp.ndarray, preds: cp.ndarray, num_classes: int) -> cp.ndarray:
    cm = cp.zeros((num_classes, num_classes), dtype=cp.int64)
    for t, p in zip(targets, preds):
        cm[int(t), int(p)] += 1
    return cm


def plot_loss_curve(history: List[Dict], path: Path) -> None:
    epochs = [h["epoch"] for h in history]
    train_losses = [h["train_loss"] for h in history]
    val_losses = [h["val_loss"] for h in history]
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, marker="o", label="Train Loss")
    plt.plot(epochs, val_losses, marker="s", label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.grid(alpha=0.3, linestyle="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def plot_confusion_matrix(cm: cp.ndarray, class_names: List[str], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    cm_np = to_numpy(cm)
    im = ax.imshow(cm_np, cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    threshold = cm_np.max() / 2 if cm_np.max() > 0 else 0
    for i in range(cm_np.shape[0]):
        for j in range(cm_np.shape[1]):
            color = "white" if cm_np[i, j] > threshold else "black"
            ax.text(j, i, str(int(cm_np[i, j])), ha="center", va="center", color=color)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_checkpoint(model: HanziCNN, class_names: List[str], img_size: int, best_val_acc: float, save_dir: Path) -> None:
    state = model.state_dict()
    cp.savez(save_dir / "best_model.npz", **state)
    meta = {
        "class_names": class_names,
        "img_size": img_size,
        "best_val_acc": best_val_acc,
    }
    with open(save_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def load_state_npz(path: Path) -> Dict[str, cp.ndarray]:
    loaded = cp.load(path, allow_pickle=False)
    return {k: loaded[k] for k in loaded.files}


def main() -> None:
    args = _ARGS
    if is_gpu:
        set_gpu_device(args.gpu_id)
        print(f"[device] Using cupy on GPU{args.gpu_id}")
    else:
        print("[device] Using numpy on CPU")
    set_seed(args.seed)

    save_dir = Path(args.save_dir).resolve()
    plot_dir = Path(args.plot_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    x, y, class_names = load_dataset(Path(args.data_dir).resolve(), args.img_size)
    num_classes = len(class_names)
    train_idx, val_idx = stratified_split(y, num_classes, args.val_ratio, args.seed)

    model = HanziCNN(num_classes=num_classes)
    criterion = CrossEntropyLoss()
    optimizer = AdamW(list(model.parameters()), lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = 0.0
    history: List[Dict] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(iter_batches(train_idx, args.batch_size, shuffle=True), total=(len(train_idx) + args.batch_size - 1) // args.batch_size, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for b in pbar:
            xb = x[b]
            yb = y[b]
            logits = model.forward(xb)
            loss = criterion.forward(logits, yb)
            grad = criterion.backward()

            optimizer.zero_grad()
            model.backward(grad)
            optimizer.step()

            preds = cp.argmax(logits, axis=1)
            train_loss_sum += loss * len(b)
            train_correct += int((preds == yb).sum().item())
            train_total += len(b)
            pbar.set_postfix(loss=f"{loss:.4f}", acc=f"{train_correct / max(train_total, 1):.4f}")

        train_loss = train_loss_sum / max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)
        val_loss, val_acc, _, _ = evaluate(model, criterion, x, y, val_idx, args.batch_size)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": args.lr,
            }
        )
        print(
            f"[Epoch {epoch:03d}] train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, class_names, args.img_size, best_val_acc, save_dir)

    with open(save_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    with open(save_dir / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)

    state = load_state_npz(save_dir / "best_model.npz")
    model.load_state_dict(state)
    _, _, tgts, preds = evaluate(model, criterion, x, y, val_idx, args.batch_size)
    cm = build_confusion_matrix(tgts, preds, num_classes)

    loss_png = plot_dir / "loss_curve.png"
    cm_png = plot_dir / "confusion_matrix.png"
    plot_loss_curve(history, loss_png)
    plot_confusion_matrix(cm, class_names, cm_png)

    print(f"Training done. Best val acc: {best_val_acc:.4f}")
    print(f"Saved ckpt dir: {save_dir}")
    print(f"Saved plots: {loss_png}, {cm_png}")


if __name__ == "__main__":
    main()
