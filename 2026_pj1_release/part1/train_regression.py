"""Train regression model y = sin(x) using the manual NN."""

from __future__ import annotations

import argparse
import os
from datetime import datetime

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from .nn import NeuralNetwork, init_backend
    from .train_classification import snapshot_model_state
except ImportError:
    from nn import NeuralNetwork, init_backend
    from train_classification import snapshot_model_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regression training for y=sin(x).")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--val-size", type=int, default=1000)
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
    epochs = np.arange(1, len(train_maes) + 1)
    plt.figure(figsize=(9, 5.5))
    plt.plot(epochs, train_maes, color="#2166ac", linewidth=2, label="train_mae")
    plt.plot(epochs, val_maes, color="#d2503c", linewidth=2, label="val_mae")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.title("Regression MAE Curves: y = sin(x)")
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    use_cuda = not args.no_cuda
    xp, to_cpu, rng = init_backend(use_cuda=use_cuda, gpu_id=args.gpu, seed=args.seed)
    hidden_sizes = [int(v) for v in args.hidden.split(",") if v.strip()]
    layer_sizes = [1, *hidden_sizes, 1]
    model = NeuralNetwork(
        layer_sizes=layer_sizes,
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

    # Save checkpoint so that infer_regression.py can reload this model.
    ckpt_path = os.path.join(args.output_dir, f"regression_best_{timestamp}.npz")
    state = snapshot_model_state(model, to_cpu)
    np.savez_compressed(
        ckpt_path,
        layer_sizes=np.asarray(layer_sizes, dtype=np.int64),
        hidden_activation=np.asarray(args.activation),
        output_activation=np.asarray("linear"),
        use_batchnorm=np.asarray(bool(args.batchnorm)),
        dropout=np.asarray(float(args.dropout), dtype=np.float32),
        **state,
    )

    print(f"Training finished. Final train MAE: {train_mae_history[-1]:.6f}")
    print(f"Training finished. Final val MAE: {val_mae_history[-1]:.6f}")
    print(f"Saved figure: {fig_path}")
    print(f"Saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
