"""Batch regression inference: load a saved ckpt and evaluate on a test set.

Inputs
------
1. ``--ckpt path/to/regression_best_*.npz``  (produced by train_regression.py)
2. ``--test-file path/to/test.npz``           (must contain arrays ``x``, ``y``)
   OR ``--test-samples N --test-range -pi,pi`` to synthesise a test set of
   ``y = sin(x)`` on the fly.

Outputs
-------
- MAE / MSE / RMSE on the test set (printed)
- A figure comparing prediction vs ground truth saved under ``--output-dir``

Note
----
Confusion matrix does not apply to scalar regression. Regression metrics
(MAE / MSE / RMSE) and a pred-vs-true plot are used instead.
"""

from __future__ import annotations

import argparse
import math
import os
from datetime import datetime
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from .nn import NeuralNetwork, init_backend
except ImportError:
    from nn import NeuralNetwork, init_backend


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regression inference: run a saved regression_best_*.npz on a test set."
    )
    parser.add_argument("--ckpt", type=str, required=True, help="Path to regression_best_*.npz.")
    parser.add_argument(
        "--test-file",
        type=str,
        default="",
        help="Path to an .npz with arrays 'x' (N,) or (N,1) and 'y' (N,) or (N,1). "
        "If empty, a synthetic test set y=sin(x) is generated.",
    )
    parser.add_argument("--test-samples", type=int, default=2000, help="Used only when --test-file is empty.")
    parser.add_argument(
        "--test-range",
        type=str,
        default=f"{-math.pi},{math.pi}",
        help="Low,high bounds for synthetic uniform sampling of x.",
    )
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--gpu", type=int, default=4)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=os.path.dirname(__file__))
    parser.add_argument("--no-figure", action="store_true", help="Skip writing the pred-vs-true PNG.")
    return parser.parse_args()


def _as_col(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2 or arr.shape[1] != 1:
        raise ValueError(f"Expected shape (N,) or (N,1), got {arr.shape}")
    return arr


def load_test_set(args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, str]:
    if args.test_file:
        path = os.path.abspath(args.test_file)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Test file not found: {path}")
        data = np.load(path, allow_pickle=False)
        if "x" not in data.files or "y" not in data.files:
            raise RuntimeError(
                f"Test npz must contain arrays 'x' and 'y'. Found: {data.files}"
            )
        x = _as_col(data["x"])
        y = _as_col(data["y"])
        if x.shape[0] != y.shape[0]:
            raise RuntimeError(f"x ({x.shape[0]}) and y ({y.shape[0]}) length mismatch.")
        return x, y, f"file: {path}"

    lo_s, hi_s = args.test_range.split(",")
    lo, hi = float(lo_s), float(hi_s)
    rng = np.random.default_rng(args.seed)
    x = rng.uniform(lo, hi, size=(args.test_samples, 1)).astype(np.float32)
    y = np.sin(x).astype(np.float32)
    return x, y, f"synthetic (N={args.test_samples}, range=[{lo:.3f},{hi:.3f}])"


def load_ckpt(path: str) -> dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    raw = np.load(path, allow_pickle=True)
    required = {
        "weights",
        "biases",
        "bn_gamma",
        "bn_beta",
        "bn_running_mean",
        "bn_running_var",
        "layer_sizes",
        "hidden_activation",
        "output_activation",
        "use_batchnorm",
    }
    missing = required - set(raw.files)
    if missing:
        raise RuntimeError(
            f"Checkpoint missing keys: {sorted(missing)}. "
            "Regenerate via the updated train_regression.py."
        )
    return {k: raw[k] for k in raw.files}


def build_model_from_ckpt(ckpt: dict, xp, rng) -> NeuralNetwork:
    layer_sizes = [int(s) for s in np.asarray(ckpt["layer_sizes"]).tolist()]
    hidden_activation = str(np.asarray(ckpt["hidden_activation"]).item())
    output_activation = str(np.asarray(ckpt["output_activation"]).item())
    use_batchnorm = bool(np.asarray(ckpt["use_batchnorm"]).item())

    model = NeuralNetwork(
        layer_sizes=layer_sizes,
        hidden_activation=hidden_activation,
        output_activation=output_activation,
        xp=xp,
        rng=rng,
        use_batchnorm=use_batchnorm,
        dropout=0.0,
        optimizer="sgd",
    )
    model.weights = [xp.asarray(w, dtype=xp.float32) for w in ckpt["weights"]]
    model.biases = [xp.asarray(b, dtype=xp.float32) for b in ckpt["biases"]]
    model.bn_gamma = [xp.asarray(g, dtype=xp.float32) for g in ckpt["bn_gamma"]]
    model.bn_beta = [xp.asarray(b, dtype=xp.float32) for b in ckpt["bn_beta"]]
    model.bn_running_mean = [xp.asarray(m, dtype=xp.float32) for m in ckpt["bn_running_mean"]]
    model.bn_running_var = [xp.asarray(v, dtype=xp.float32) for v in ckpt["bn_running_var"]]
    return model


def batched_forward(model: NeuralNetwork, x_np: np.ndarray, batch_size: int, xp, to_cpu) -> np.ndarray:
    parts: List[np.ndarray] = []
    n = x_np.shape[0]
    for start in range(0, n, batch_size):
        xb = xp.asarray(x_np[start : start + batch_size], dtype=xp.float32)
        out = model.forward(xb, training=False)
        parts.append(np.asarray(to_cpu(out)))
    return np.concatenate(parts, axis=0) if parts else np.zeros((0, 1), dtype=np.float32)


def save_pred_vs_true(x: np.ndarray, y: np.ndarray, y_pred: np.ndarray, out_path: str, metrics: dict) -> None:
    order = np.argsort(x.reshape(-1))
    x_sorted = x.reshape(-1)[order]
    y_sorted = y.reshape(-1)[order]
    yhat_sorted = y_pred.reshape(-1)[order]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(x_sorted, y_sorted, color="#2166ac", linewidth=1.5, label="True")
    axes[0].plot(x_sorted, yhat_sorted, color="#d2503c", linewidth=1.2, linestyle="--", label="Pred")
    axes[0].set_title(
        f"Prediction vs True  (MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f})"
    )
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].grid(alpha=0.3, linestyle="--")
    axes[0].legend()

    resid = yhat_sorted - y_sorted
    axes[1].scatter(x_sorted, resid, s=6, alpha=0.55, color="#888")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_title("Residual (Pred - True)")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("residual")
    axes[1].grid(alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    use_cuda = not args.no_cuda
    xp, to_cpu, rng = init_backend(use_cuda=use_cuda, gpu_id=args.gpu, seed=args.seed)

    ckpt = load_ckpt(args.ckpt)
    model = build_model_from_ckpt(ckpt, xp=xp, rng=rng)

    x_test, y_test, subset_desc = load_test_set(args)
    y_pred = batched_forward(model, x_test, args.batch_size, xp=xp, to_cpu=to_cpu)

    diff = y_pred - y_test
    mse = float(np.mean(diff * diff))
    mae = float(np.mean(np.abs(diff)))
    rmse = float(math.sqrt(mse))
    metrics = {"mae": mae, "mse": mse, "rmse": rmse}

    layer_sizes = [int(s) for s in np.asarray(ckpt["layer_sizes"]).tolist()]
    hidden_activation = str(np.asarray(ckpt["hidden_activation"]).item())
    use_bn = bool(np.asarray(ckpt["use_batchnorm"]).item())

    print(f"Checkpoint : {args.ckpt}")
    print(f"Test set   : {subset_desc}")
    print(f"Num samples: {x_test.shape[0]}")
    print(f"Activation : {hidden_activation}  |  BatchNorm: {use_bn}")
    print(f"Layer sizes: {layer_sizes}")
    print(f"MAE        : {mae:.6f}")
    print(f"MSE        : {mse:.6f}")
    print(f"RMSE       : {rmse:.6f}")
    print("Note: confusion matrix is not applicable to scalar regression.")

    if not args.no_figure:
        os.makedirs(args.output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fig_path = os.path.join(args.output_dir, f"regression_infer_pred_{ts}.png")
        save_pred_vs_true(x_test, y_test, y_pred, fig_path, metrics)
        print(f"Saved figure: {fig_path}")


if __name__ == "__main__":
    main()
