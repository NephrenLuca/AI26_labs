import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from models import HanziCNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CNN for handwritten Hanzi classification.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="../train",
        help="Path to dataset root. Expected ImageFolder layout: class_name/*.bmp",
    )
    parser.add_argument("--save-dir", type=str, default="./checkpoints", help="Directory to save outputs.")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size per iteration.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for optimizer.")
    parser.add_argument("--img-size", type=int, default=64, help="Resize input images to img-size x img-size.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker count.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device type.")
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default="0",
        help="Comma-separated CUDA ids, e.g. '4,5,6,7'. For single GPU: '4'.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_gpu_ids(gpu_ids_str: str) -> List[int]:
    return [int(x.strip()) for x in gpu_ids_str.split(",") if x.strip()]


def select_device(device_type: str, gpu_ids: List[int]) -> Tuple[torch.device, List[int]]:
    if device_type == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu"), []

    visible_gpu_count = torch.cuda.device_count()
    valid_ids = [idx for idx in gpu_ids if 0 <= idx < visible_gpu_count]
    if not valid_ids:
        valid_ids = [0]
    torch.cuda.set_device(valid_ids[0])
    return torch.device(f"cuda:{valid_ids[0]}"), valid_ids


def stratified_split(dataset: datasets.ImageFolder, val_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    rng = random.Random(seed)
    class_to_indices: Dict[int, List[int]] = {i: [] for i in range(len(dataset.classes))}
    for idx, (_, target) in enumerate(dataset.samples):
        class_to_indices[target].append(idx)

    train_indices: List[int] = []
    val_indices: List[int] = []
    for _, indices in class_to_indices.items():
        rng.shuffle(indices)
        split_point = max(1, int(len(indices) * val_ratio))
        val_indices.extend(indices[:split_point])
        train_indices.extend(indices[split_point:])

    return train_indices, val_indices


def build_dataloaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader, List[str]]:
    data_dir = Path(args.data_dir).resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    train_transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((args.img_size, args.img_size)),
            transforms.RandomAffine(degrees=8, translate=(0.08, 0.08), scale=(0.92, 1.08)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    base_dataset = datasets.ImageFolder(str(data_dir))
    train_idx, val_idx = stratified_split(base_dataset, args.val_ratio, args.seed)

    train_dataset_full = datasets.ImageFolder(str(data_dir), transform=train_transform)
    val_dataset_full = datasets.ImageFolder(str(data_dir), transform=eval_transform)

    train_dataset = Subset(train_dataset_full, train_idx)
    val_dataset = Subset(val_dataset_full, val_idx)

    pin_memory = args.device == "cuda" and torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return train_loader, val_loader, base_dataset.classes


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, targets)

            running_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

    avg_loss = running_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


def evaluate_with_predictions(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_targets: List[np.ndarray] = []
    all_preds: List[np.ndarray] = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, targets)
            preds = logits.argmax(dim=1)

            running_loss += loss.item() * images.size(0)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

            all_targets.append(targets.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    avg_loss = running_loss / max(total, 1)
    acc = correct / max(total, 1)
    targets_np = np.concatenate(all_targets, axis=0) if all_targets else np.array([])
    preds_np = np.concatenate(all_preds, axis=0) if all_preds else np.array([])
    return avg_loss, acc, targets_np, preds_np


def build_confusion_matrix(targets: np.ndarray, preds: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(targets, preds):
        cm[int(t), int(p)] += 1
    return cm


def plot_loss_curve(history: List[Dict], save_path: Path) -> None:
    epochs = [item["epoch"] for item in history]
    train_losses = [item["train_loss"] for item in history]
    val_losses = [item["val_loss"] for item in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train Loss", marker="o", linewidth=1.5)
    plt.plot(epochs, val_losses, label="Val Loss", marker="s", linewidth=1.5)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.grid(alpha=0.3, linestyle="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], save_path: Path) -> None:
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
        title="Confusion Matrix",
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
    set_seed(args.seed)
    save_dir = Path(args.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    gpu_ids = parse_gpu_ids(args.gpu_ids)
    device, valid_gpu_ids = select_device(args.device, gpu_ids)
    print(f"Using device: {device}")
    if valid_gpu_ids:
        print(f"Using GPU IDs: {valid_gpu_ids}")

    train_loader, val_loader, class_names = build_dataloaders(args)
    num_classes = len(class_names)
    if num_classes != 12:
        print(f"Warning: expected 12 classes but found {num_classes}")

    model = HanziCNN(num_classes=num_classes)
    model = model.to(device)

    if device.type == "cuda" and len(valid_gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=valid_gpu_ids, output_device=valid_gpu_ids[0])
        print("Enabled nn.DataParallel for multi-GPU training.")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for images, targets in pbar:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            train_correct += (preds == targets).sum().item()
            train_total += targets.size(0)

            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{train_correct / max(train_total, 1):.4f}")

        scheduler.step()

        train_loss = train_loss_sum / max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        epoch_log = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_log)
        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, lr={epoch_log['lr']:.6f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(
                {
                    "model_state_dict": state_dict,
                    "class_names": class_names,
                    "img_size": args.img_size,
                    "best_val_acc": best_val_acc,
                },
                save_dir / "best_model.pt",
            )
            print(f"Saved new best model with val_acc={best_val_acc:.4f}")

    with open(save_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    with open(save_dir / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)

    # Save plots to current working directory as requested.
    output_dir = Path.cwd()
    loss_plot_path = output_dir / "loss_curve.png"
    plot_loss_curve(history, loss_plot_path)

    best_ckpt = torch.load(save_dir / "best_model.pt", map_location=device)
    best_state_dict = best_ckpt["model_state_dict"]
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(best_state_dict)
    else:
        model.load_state_dict(best_state_dict)

    _, _, targets_np, preds_np = evaluate_with_predictions(model, val_loader, criterion, device)
    cm = build_confusion_matrix(targets_np, preds_np, num_classes=len(class_names))
    cm_plot_path = output_dir / "confusion_matrix.png"
    plot_confusion_matrix(cm, class_names, cm_plot_path)

    print(f"Training complete. Best val acc: {best_val_acc:.4f}")
    print(f"Artifacts saved to: {save_dir}")
    print(f"Saved loss curve: {loss_plot_path}")
    print(f"Saved confusion matrix: {cm_plot_path}")


if __name__ == "__main__":
    main()
