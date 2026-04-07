"""Train 12-class handwritten character classifier using the manual NN."""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw

try:
    from .nn import NeuralNetwork, init_backend
except ImportError:
    from nn import NeuralNetwork, init_backend


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classification training for handwritten characters.")
    parser.add_argument("--data-dir", type=str, default=os.path.join(os.path.dirname(__file__), "train"))
    parser.add_argument("--img-size", type=str, default="28,28")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--hidden", type=str, default="256,128")
    parser.add_argument("--activation", type=str, default="relu", choices=["relu", "tanh", "sigmoid"])
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batchnorm", action="store_true")
    parser.add_argument("--bn-momentum", type=float, default=0.9)
    parser.add_argument("--bn-eps", type=float, default=1e-5)
    parser.add_argument("--gpu", type=int, default=4)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=os.path.dirname(__file__))
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--aug-prob", type=float, default=0.8)
    parser.add_argument("--aug-rotate", type=float, default=12.0)
    parser.add_argument("--aug-translate", type=float, default=0.08)
    parser.add_argument("--aug-scale-min", type=float, default=0.92)
    parser.add_argument("--aug-scale-max", type=float, default=1.10)
    parser.add_argument("--aug-noise-std", type=float, default=0.03)
    return parser.parse_args()


def _parse_hw(s: str) -> Tuple[int, int]:
    h, w = s.split(",")
    return int(h), int(w)


def load_image_dataset(data_dir: str, image_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    class_names = sorted([n for n in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, n))])
    if len(class_names) == 0:
        raise RuntimeError(f"No class folders found in {data_dir}")

    xs: List[np.ndarray] = []
    ys: List[int] = []
    valid_ext = {".bmp", ".png", ".jpg", ".jpeg"}

    for label, cls in enumerate(class_names):
        cls_dir = os.path.join(data_dir, cls)
        for name in os.listdir(cls_dir):
            ext = os.path.splitext(name)[1].lower()
            if ext not in valid_ext:
                continue
            path = os.path.join(cls_dir, name)
            with Image.open(path) as img:
                gray = img.convert("L").resize((image_size[1], image_size[0]))
                arr = np.asarray(gray, dtype=np.float32) / 255.0
            xs.append(arr.reshape(-1))
            ys.append(label)

    if len(xs) == 0:
        raise RuntimeError(f"No image files found in class folders under {data_dir}")

    x = np.stack(xs, axis=0).astype(np.float32)
    y = np.asarray(ys, dtype=np.int64)
    return x, y, class_names


def train_val_split(x: np.ndarray, y: np.ndarray, val_ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    idx = np.arange(x.shape[0])
    rng.shuffle(idx)
    split = int((1.0 - val_ratio) * len(idx))
    train_idx = idx[:split]
    val_idx = idx[split:]
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


def _scale_and_center(img: Image.Image, scale: float, target_hw: Tuple[int, int]) -> Image.Image:
    h, w = target_hw
    scaled_h = max(1, int(round(h * scale)))
    scaled_w = max(1, int(round(w * scale)))
    resized = img.resize((scaled_w, scaled_h), resample=Image.BILINEAR)
    if scale >= 1.0:
        left = (scaled_w - w) // 2
        top = (scaled_h - h) // 2
        return resized.crop((left, top, left + w, top + h))
    canvas = Image.new("L", (w, h), 0)
    left = (w - scaled_w) // 2
    top = (h - scaled_h) // 2
    canvas.paste(resized, (left, top))
    return canvas


def augment_batch(
    x_batch: np.ndarray,
    image_size: Tuple[int, int],
    rng: np.random.Generator,
    prob: float,
    max_rotate_deg: float,
    max_translate_ratio: float,
    scale_min: float,
    scale_max: float,
    noise_std: float,
) -> np.ndarray:
    h, w = image_size
    aug = np.empty_like(x_batch, dtype=np.float32)
    max_dx = int(round(max_translate_ratio * w))
    max_dy = int(round(max_translate_ratio * h))
    for i, flat in enumerate(x_batch):
        arr = np.clip(flat.reshape(h, w), 0.0, 1.0)
        if rng.random() >= prob:
            aug[i] = arr.reshape(-1).astype(np.float32)
            continue

        img = Image.fromarray((arr * 255.0).astype(np.uint8), mode="L")
        scale = float(rng.uniform(scale_min, scale_max))
        img = _scale_and_center(img, scale=scale, target_hw=(h, w))

        angle = float(rng.uniform(-max_rotate_deg, max_rotate_deg))
        img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=0)

        tx = int(rng.integers(-max_dx, max_dx + 1)) if max_dx > 0 else 0
        ty = int(rng.integers(-max_dy, max_dy + 1)) if max_dy > 0 else 0
        img = img.transform((w, h), Image.AFFINE, (1, 0, tx, 0, 1, ty), resample=Image.BILINEAR, fillcolor=0)

        out = np.asarray(img, dtype=np.float32) / 255.0
        if noise_std > 0.0:
            out = out + rng.normal(loc=0.0, scale=noise_std, size=out.shape).astype(np.float32)
        out = np.clip(out, 0.0, 1.0)
        aug[i] = out.reshape(-1).astype(np.float32)
    return aug


def softmax_ce_loss(logits, targets, xp) -> tuple[float, object]:
    shifted = logits - xp.max(logits, axis=1, keepdims=True)
    exp = xp.exp(shifted)
    probs = exp / xp.sum(exp, axis=1, keepdims=True)
    n = logits.shape[0]
    loss = -xp.mean(xp.log(xp.clip(probs[xp.arange(n), targets], 1e-12, 1.0)))
    grad = probs
    grad[xp.arange(n), targets] -= 1.0
    grad /= n
    return float(loss.item()), grad


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def save_confusion_matrix_image(cm: np.ndarray, class_names: List[str], val_acc: float, out_path: str) -> None:
    n = len(class_names)
    cell = 46
    left, top = 130, 90
    width = left + n * cell + 40
    height = top + n * cell + 70
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    max_value = int(cm.max()) if cm.size > 0 else 1
    if max_value <= 0:
        max_value = 1

    draw.text((20, 20), f"Confusion Matrix (val_acc={val_acc:.4f})", fill="black")
    draw.text((width // 2 - 45, height - 25), "Predicted", fill="black")
    draw.text((20, top + n * cell // 2), "True", fill="black")

    for i, name in enumerate(class_names):
        draw.text((left + i * cell + 6, top - 20), str(name), fill="black")
        draw.text((left - 40, top + i * cell + 14), str(name), fill="black")

    for i in range(n):
        for j in range(n):
            value = int(cm[i, j])
            intensity = int(255 - (value / max_value) * 180)
            color = (intensity, intensity, 255)
            x0 = left + j * cell
            y0 = top + i * cell
            x1 = x0 + cell
            y1 = y0 + cell
            draw.rectangle((x0, y0, x1, y1), fill=color, outline="gray")
            text_color = "black" if value < (max_value * 0.5) else "white"
            draw.text((x0 + 12, y0 + 14), str(value), fill=text_color)

    img.save(out_path)


def main() -> None:
    args = parse_args()
    if not (0.0 <= args.aug_prob <= 1.0):
        raise ValueError("--aug-prob must be in [0, 1].")
    if args.aug_translate < 0.0:
        raise ValueError("--aug-translate must be >= 0.")
    if args.aug_scale_min <= 0.0 or args.aug_scale_max <= 0.0:
        raise ValueError("--aug-scale-min/--aug-scale-max must be > 0.")
    if args.aug_scale_min > args.aug_scale_max:
        raise ValueError("--aug-scale-min must be <= --aug-scale-max.")
    if args.aug_noise_std < 0.0:
        raise ValueError("--aug-noise-std must be >= 0.")

    use_cuda = not args.no_cuda
    xp, to_cpu, rng = init_backend(use_cuda=use_cuda, gpu_id=args.gpu, seed=args.seed)
    np_rng = np.random.default_rng(args.seed)
    h, w = _parse_hw(args.img_size)
    x, y, class_names = load_image_dataset(args.data_dir, image_size=(h, w))

    x_train, y_train, x_val, y_val = train_val_split(x, y, args.val_ratio, args.seed)
    if len(y_val) == 0:
        raise RuntimeError("Validation set is empty. Increase dataset size or val-ratio.")

    hidden_sizes = [int(v) for v in args.hidden.split(",") if v.strip()]
    model = NeuralNetwork(
        layer_sizes=[x_train.shape[1], *hidden_sizes, len(class_names)],
        hidden_activation=args.activation,
        output_activation="linear",
        seed=args.seed,
        xp=xp,
        rng=rng,
        use_batchnorm=args.batchnorm,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps,
        dropout=args.dropout,
        optimizer=args.optimizer,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_eps=args.adam_eps,
    )

    x_val_xp = xp.asarray(x_val, dtype=xp.float32)
    y_val_xp = xp.asarray(y_val, dtype=xp.int64)

    n_train = x_train.shape[0]
    for _ in range(args.epochs):
        idx = np.arange(n_train)
        np_rng.shuffle(idx)
        for start in range(0, n_train, args.batch_size):
            b = idx[start : start + args.batch_size]
            xb_np = x_train[b]
            if not args.no_augment:
                xb_np = augment_batch(
                    xb_np,
                    image_size=(h, w),
                    rng=np_rng,
                    prob=args.aug_prob,
                    max_rotate_deg=args.aug_rotate,
                    max_translate_ratio=args.aug_translate,
                    scale_min=args.aug_scale_min,
                    scale_max=args.aug_scale_max,
                    noise_std=args.aug_noise_std,
                )
            xb = xp.asarray(xb_np, dtype=xp.float32)
            yb = xp.asarray(y_train[b], dtype=xp.int64)
            logits = model.forward(xb, training=True)
            _, grad = softmax_ce_loss(logits, yb, xp)
            model.backward(grad, lr=args.lr)

    val_logits = model.forward(x_val_xp, training=False)
    val_pred = xp.argmax(val_logits, axis=1)
    val_acc = float(xp.mean((val_pred == y_val_xp).astype(xp.float32)).item())
    cm = confusion_matrix(to_cpu(y_val_xp), to_cpu(val_pred), num_classes=len(class_names))

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_path = os.path.join(args.output_dir, f"classification_confusion_{timestamp}.png")

    save_confusion_matrix_image(cm, class_names, val_acc, fig_path)

    print(f"Training finished. Validation accuracy: {val_acc:.4f}")
    print(f"Saved figure: {fig_path}")


if __name__ == "__main__":
    main()
