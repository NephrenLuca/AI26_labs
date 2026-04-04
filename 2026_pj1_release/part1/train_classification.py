"""
训练入口：分类任务 — 12 类手写汉字（flatten + MLP，课程 Part 1）。

数据目录：每类一个子文件夹，字典序对应标签 ``0 .. K-1``（见 ``data_chars.load_char_dataset``）。

运行示例（项目根目录）::

    python -m part1.train_classification --data path/to/train_root
    python -m part1.train_classification --data path/to/train --gpu 7
    python -m part1.train_classification --synthetic --epochs 5

真实数据请先按 README 组织 ``0/`` … ``11/``（或 12 个任意名称的子文件夹）。
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from . import backend
from .data_chars import load_char_dataset, make_synthetic_char_dataset
from .hyperparams import CHAR_DEFAULTS
from .losses import CrossEntropyLoss
from .network import MLP
from .optimizer import SGDOptimizer, make_lr_fn
from .utils import clip_tensor_l2_norm, set_random_seed


def build_arg_parser() -> argparse.ArgumentParser:
    d = CHAR_DEFAULTS
    p = argparse.ArgumentParser(
        description="Part 1 classification: 12-class characters with manual backprop MLP."
    )
    p.add_argument(
        "--data",
        type=str,
        default="",
        help="Root folder with one subdirectory per class (ignored if --synthetic).",
    )
    p.add_argument(
        "--synthetic",
        action="store_true",
        help="Use random noise images to smoke-test the training loop.",
    )
    p.add_argument(
        "--gpu",
        type=int,
        default=None,
        metavar="ID",
        help="Physical CUDA device index (e.g. 4–7).",
    )
    p.add_argument("--no-cuda", action="store_true", help="Force NumPy CPU.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--hidden",
        type=str,
        default=",".join(str(x) for x in d.hidden_sizes),
        help="Comma-separated hidden layer sizes, e.g. 512,256,128",
    )
    p.add_argument(
        "--activation",
        choices=("relu", "tanh"),
        default=d.hidden_activation,
    )
    p.add_argument("--epochs", type=int, default=d.epochs)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument(
        "--lr-schedule",
        choices=("constant", "step", "cosine"),
        default=d.lr_schedule,
    )
    p.add_argument("--lr-step-size", type=int, default=d.lr_step_size)
    p.add_argument("--lr-gamma", type=float, default=d.lr_gamma)
    p.add_argument("--lr-eta-min", type=float, default=d.lr_eta_min)
    p.add_argument("--momentum", type=float, default=d.momentum)
    p.add_argument("--weight-decay", type=float, default=d.weight_decay)
    p.add_argument(
        "--grad-clip",
        type=float,
        default=d.grad_clip if d.grad_clip is not None else -1.0,
        help="L2 clip on loss gradient; negative to disable.",
    )
    p.add_argument("--num-classes", type=int, default=d.num_classes)
    p.add_argument(
        "--img-size",
        type=str,
        default="28,28",
        help="H,W for resize (synthetic and load).",
    )
    p.add_argument(
        "--val-ratio",
        type=float,
        default=0.0,
        help="Fraction of data for validation (0 = train on all, report train acc).",
    )
    p.add_argument("--log-every", type=int, default=1, help="Print metrics every N epochs.")
    return p


def _parse_hidden(s: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in s.split(",") if x.strip())


def _parse_img_size(s: str) -> tuple[int, int]:
    parts = [x.strip() for x in s.split(",") if x.strip()]
    if len(parts) != 2:
        raise ValueError("--img-size expects H,W")
    return int(parts[0]), int(parts[1])


def _train_val_split(
    X: np.ndarray,
    y: np.ndarray,
    val_ratio: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = X.shape[0]
    idx = rng.permutation(n)
    n_val = int(round(n * val_ratio))
    n_val = min(max(n_val, 0), n - 1) if n > 1 else 0
    v_idx = idx[:n_val]
    t_idx = idx[n_val:]
    return X[t_idx], y[t_idx], X[v_idx], y[v_idx]


def _accuracy_batches(
    net: MLP,
    X,
    y,
    *,
    batch_size: int,
    xp,
) -> float:
    """Classification accuracy in [0, 1]."""
    n = int(X.shape[0])
    if n == 0:
        return float("nan")
    correct = 0
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        sl = slice(start, end)
        logits = net.forward(X[sl])
        pred = xp.argmax(logits, axis=1)
        yt = y[sl]
        correct += int(backend.to_cpu_array(xp.sum(pred == yt)))
    return correct / n


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    grad_clip = None if args.grad_clip < 0 else args.grad_clip
    hidden = _parse_hidden(args.hidden)
    img_size = _parse_img_size(args.img_size)

    rng = np.random.default_rng(args.seed)

    if args.synthetic:
        X_np, y_np, names = make_synthetic_char_dataset(
            n_per_class=200,
            image_size=img_size,
            num_classes=args.num_classes,
            seed=args.seed,
        )
    else:
        if not args.data:
            print("Error: provide --data DIR or use --synthetic.", file=sys.stderr)
            return 2
        X_np, y_np, names = load_char_dataset(
            args.data,
            image_size=img_size,
            num_classes=args.num_classes,
        )

    in_dim = X_np.shape[1]
    layer_sizes = [in_dim] + list(hidden) + [args.num_classes]

    X_tr, y_tr, X_va, y_va = _train_val_split(X_np, y_np, args.val_ratio, rng)

    backend.init_backend(use_cuda=not args.no_cuda, gpu_id=args.gpu)
    xp = backend.get_xp()
    set_random_seed(args.seed)

    X_tr_d = xp.asarray(X_tr)
    y_tr_d = xp.asarray(y_tr)
    if X_va.shape[0] > 0:
        X_va_d = xp.asarray(X_va)
        y_va_d = xp.asarray(y_va)
    else:
        X_va_d = None
        y_va_d = None

    net = MLP(layer_sizes, hidden_activation=args.activation)
    criterion = CrossEntropyLoss()
    opt = SGDOptimizer(
        net.linear_layers(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    lr_fn = make_lr_fn(
        args.lr_schedule,
        args.lr,
        args.epochs,
        step_size=args.lr_step_size,
        gamma=args.lr_gamma,
        eta_min=args.lr_eta_min,
    )

    n = int(X_tr_d.shape[0])
    batches = max(1, (n + args.batch_size - 1) // args.batch_size)
    mode = f"CuPy GPU {backend.device_id()}" if backend.is_cuda() else "NumPy CPU"
    print(
        f"Backend: {mode} | classes={args.num_classes} ({names[:3]}{'...' if len(names) > 3 else ''}) "
        f"| N_train={n} | layers={layer_sizes}"
    )

    for epoch in range(args.epochs):
        opt.lr = lr_fn(epoch)
        idx = rng.permutation(n)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n, args.batch_size):
            end = min(start + args.batch_size, n)
            bi = xp.asarray(idx[start:end], dtype=xp.int32)
            xb = X_tr_d[bi]
            yb = y_tr_d[bi]
            logits = net.forward(xb)
            loss = criterion.forward(logits, yb)
            epoch_loss += float(backend.to_cpu_array(loss))
            n_batches += 1
            g = criterion.backward()
            clip_tensor_l2_norm(g, grad_clip)
            net.backward(g)
            opt.step()
        epoch_loss /= max(n_batches, 1)

        if epoch % args.log_every == 0 or epoch == args.epochs - 1:
            tr_acc = _accuracy_batches(
                net, X_tr_d, y_tr_d, batch_size=args.batch_size, xp=xp
            )
            if X_va_d is not None and y_va_d is not None and X_va_d.shape[0] > 0:
                va_acc = _accuracy_batches(
                    net, X_va_d, y_va_d, batch_size=args.batch_size, xp=xp
                )
                print(
                    f"epoch {epoch:4d}  lr={opt.lr:.6g}  "
                    f"loss={epoch_loss:.5f}  train_acc={tr_acc:.4f}  val_acc={va_acc:.4f}"
                )
            else:
                print(
                    f"epoch {epoch:4d}  lr={opt.lr:.6g}  "
                    f"loss={epoch_loss:.5f}  train_acc={tr_acc:.4f}"
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
