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
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--hidden", type=str, default="64,64")
    parser.add_argument("--activation", type=str, default="tanh", choices=["relu", "tanh", "sigmoid"])
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batchnorm", action="store_true")
    parser.add_argument("--bn-momentum", type=float, default=0.9)
    parser.add_argument("--bn-eps", type=float, default=1e-5)
    parser.add_argument("--gpu", type=int, default=4)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=os.path.dirname(__file__))
    return parser.parse_args()


def mse_loss(pred, target, xp) -> tuple[float, object]:
    diff = pred - target
    loss = float(np.asarray(xp.mean(diff * diff)))
    grad = 2.0 * diff / diff.shape[0]
    return loss, grad


def save_loss_curve(losses: list[float], out_path: str) -> None:
    width, height = 1000, 600
    margin_left, margin_right = 80, 30
    margin_top, margin_bottom = 50, 70

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    x0, y0 = margin_left, height - margin_bottom
    x1, y1 = width - margin_right, margin_top
    draw.line((x0, y0, x1, y0), fill="black", width=2)
    draw.line((x0, y0, x0, y1), fill="black", width=2)
    draw.text((x0 + 5, y1 - 28), "Loss", fill="black")
    draw.text((x1 - 40, y0 + 8), "Epoch", fill="black")
    draw.text((width // 2 - 150, 12), "Regression Training Curve: y = sin(x)", fill="black")

    if len(losses) == 0:
        img.save(out_path)
        return

    y_min = min(losses)
    y_max = max(losses)
    if abs(y_max - y_min) < 1e-12:
        y_max = y_min + 1e-12

    points = []
    n = len(losses)
    for i, loss in enumerate(losses):
        px = x0 + (x1 - x0) * (i / max(1, n - 1))
        py = y0 - (y0 - y1) * ((loss - y_min) / (y_max - y_min))
        points.append((px, py))

    if len(points) >= 2:
        draw.line(points, fill=(33, 102, 172), width=2)
    else:
        px, py = points[0]
        draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill=(33, 102, 172))

    draw.text((x0, y0 + 8), "1", fill="black")
    draw.text((x1 - 15, y0 + 8), str(n), fill="black")
    draw.text((8, y0 - 6), f"{y_min:.4f}", fill="black")
    draw.text((8, y1 - 6), f"{y_max:.4f}", fill="black")
    draw.text((x0 + 10, y1 + 8), f"Final loss: {losses[-1]:.6f}", fill="black")
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
    )

    history = []
    for _ in range(args.epochs):
        x = rng.uniform(-np.pi, np.pi, size=(args.batch_size, 1)).astype(xp.float32)
        y = xp.sin(x)
        pred = model.forward(x, training=True)
        loss, grad = mse_loss(pred, y, xp)
        model.backward(grad, lr=args.lr)
        history.append(loss)

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_path = os.path.join(args.output_dir, f"regression_loss_{timestamp}.png")

    save_loss_curve(history, fig_path)

    print(f"Training finished. Final train loss: {history[-1]:.6f}")
    print(f"Saved figure: {fig_path}")


if __name__ == "__main__":
    main()
