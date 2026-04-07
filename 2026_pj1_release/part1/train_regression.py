"""Train regression model y = sin(x) using the manual NN."""

from __future__ import annotations

import argparse
import os
from datetime import datetime

import numpy as np
from PIL import Image, ImageDraw

try:
    from .nn import NeuralNetwork, init_backend
except ImportError:
    from nn import NeuralNetwork, init_backend


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regression training for y=sin(x).")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--val-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--hidden", type=str, default="64,64")
    parser.add_argument("--activation", type=str, default="tanh", choices=["relu", "tanh", "sigmoid"])
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batchnorm", action="store_true", default=False)
    parser.add_argument("--bn-momentum", type=float, default=0.9)
    parser.add_argument("--bn-eps", type=float, default=1e-5)
    parser.add_argument("--gpu", type=int, default=4)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=os.path.dirname(__file__))
    return parser.parse_args()


def mse_loss(pred, target, xp) -> tuple[float, object]:
    diff = pred - target
    loss = float(xp.mean(diff * diff).item())
    grad = 2.0 * diff / diff.shape[0]
    return loss, grad


def mae_metric(pred, target, xp) -> float:
    return float(xp.mean(xp.abs(pred - target)).item())


def save_mae_curve(train_maes: list[float], val_maes: list[float], out_path: str) -> None:
    width, height = 1000, 600
    margin_left, margin_right = 80, 30
    margin_top, margin_bottom = 50, 70

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    x0, y0 = margin_left, height - margin_bottom
    x1, y1 = width - margin_right, margin_top
    draw.line((x0, y0, x1, y0), fill="black", width=2)
    draw.line((x0, y0, x0, y1), fill="black", width=2)
    draw.text((x0 + 5, y1 - 28), "MAE", fill="black")
    draw.text((x1 - 40, y0 + 8), "Epoch", fill="black")
    draw.text((width // 2 - 170, 12), "Regression MAE Curve: y = sin(x)", fill="black")

    if len(train_maes) == 0:
        img.save(out_path)
        return

    all_maes = train_maes + val_maes
    y_min = min(all_maes)
    y_max = max(all_maes)
    if abs(y_max - y_min) < 1e-12:
        y_max = y_min + 1e-12

    train_points = []
    val_points = []
    n = len(train_maes)
    for i, mae in enumerate(train_maes):
        px = x0 + (x1 - x0) * (i / max(1, n - 1))
        py = y0 - (y0 - y1) * ((mae - y_min) / (y_max - y_min))
        train_points.append((px, py))

    for i, mae in enumerate(val_maes):
        px = x0 + (x1 - x0) * (i / max(1, n - 1))
        py = y0 - (y0 - y1) * ((mae - y_min) / (y_max - y_min))
        val_points.append((px, py))

    if len(train_points) >= 2:
        draw.line(train_points, fill=(33, 102, 172), width=2)
    else:
        px, py = train_points[0]
        draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill=(33, 102, 172))
    if len(val_points) >= 2:
        draw.line(val_points, fill=(210, 80, 60), width=2)
    elif len(val_points) == 1:
        px, py = val_points[0]
        draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill=(210, 80, 60))

    draw.text((x0, y0 + 8), "1", fill="black")
    draw.text((x1 - 15, y0 + 8), str(n), fill="black")
    draw.text((8, y0 - 6), f"{y_min:.4f}", fill="black")
    draw.text((8, y1 - 6), f"{y_max:.4f}", fill="black")
    draw.text((x0 + 10, y1 + 8), f"Final train MAE: {train_maes[-1]:.6f}", fill=(33, 102, 172))
    draw.text((x0 + 10, y1 + 28), f"Final val MAE: {val_maes[-1]:.6f}", fill=(210, 80, 60))
    draw.text((x1 - 180, y1 + 8), "Blue: train_mae", fill=(33, 102, 172))
    draw.text((x1 - 180, y1 + 28), "Red: val_mae", fill=(210, 80, 60))
    img.save(out_path)


def main() -> None:
    args = parse_args()
    use_cuda = not args.no_cuda
    xp, _, rng = init_backend(use_cuda=use_cuda, gpu_id=args.gpu, seed=args.seed)
    hidden_sizes = [int(v) for v in args.hidden.split(",") if v.strip()]
    model = NeuralNetwork(
        layer_sizes=[1, *hidden_sizes, 1],
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

    x_val = rng.uniform(-np.pi, np.pi, size=(args.val_size, 1)).astype(xp.float32)
    y_val = xp.sin(x_val)

    train_mae_history = []
    val_mae_history = []
    for _ in range(args.epochs):
        x = rng.uniform(-np.pi, np.pi, size=(args.batch_size, 1)).astype(xp.float32)
        y = xp.sin(x)
        pred = model.forward(x, training=True)
        loss, grad = mse_loss(pred, y, xp)
        model.backward(grad, lr=args.lr)
        train_mae_history.append(mae_metric(pred, y, xp))
        val_pred = model.forward(x_val, training=False)
        val_mae_history.append(mae_metric(val_pred, y_val, xp))

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_path = os.path.join(args.output_dir, f"regression_mae_{timestamp}.png")

    save_mae_curve(train_mae_history, val_mae_history, fig_path)

    print(f"Training finished. Final train MAE: {train_mae_history[-1]:.6f}")
    print(f"Training finished. Final val MAE: {val_mae_history[-1]:.6f}")
    print(f"Saved figure: {fig_path}")


if __name__ == "__main__":
    main()
