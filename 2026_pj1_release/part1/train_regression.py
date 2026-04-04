"""
训练入口：回归任务 — 在 ``[-π, π]`` 上拟合 ``y = sin(x)``（课程 Part 1）。

运行示例（项目根目录）::

    python -m part1.train_regression
    python -m part1.train_regression --gpu 5
    python -m part1.train_regression --gpu 6 --epochs 6000 --layers 1,256,256,1

多卡机器上指定物理 GPU 4–7：``--gpu 4`` … ``--gpu 7``（需已安装与 CUDA 匹配的 CuPy）。
"""

from __future__ import annotations

import argparse
import math
import sys

from . import backend
from .data_sin import eval_grid_mae
from .hyperparams import SIN_DEFAULTS
from .losses import MSELoss
from .network import MLP
from .optimizer import SGDOptimizer, make_lr_fn
from .utils import clip_tensor_l2_norm, set_random_seed


def _parse_layers(s: str) -> tuple[int, ...]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return tuple(int(x) for x in parts)


def build_arg_parser() -> argparse.ArgumentParser:
    d = SIN_DEFAULTS
    p = argparse.ArgumentParser(
        description="Part 1 regression: fit sin(x) on [-pi, pi] with manual backprop MLP."
    )
    p.add_argument(
        "--gpu",
        type=int,
        default=None,
        metavar="ID",
        help="Physical CUDA device index (e.g. 4–7). Default: CuPy device 0, or CPU with --no-cuda.",
    )
    p.add_argument(
        "--no-cuda",
        action="store_true",
        help="Force NumPy on CPU (ignore --gpu).",
    )
    p.add_argument("--seed", type=int, default=42, help="RNG seed (Python + NumPy/CuPy).")
    p.add_argument(
        "--layers",
        type=str,
        default=",".join(str(x) for x in d.layer_sizes),
        help="Comma-separated layer widths, e.g. 1,128,128,1",
    )
    p.add_argument(
        "--activation",
        choices=("tanh", "relu"),
        default=d.hidden_activation,
        help="Hidden activation (output layer is always linear).",
    )
    p.add_argument("--epochs", type=int, default=d.epochs)
    p.add_argument("--samples-per-epoch", type=int, default=d.samples_per_epoch)
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
        help="L2 clip on loss gradient; use negative value to disable.",
    )
    p.add_argument("--log-every", type=int, default=200, help="Print MAE every N epochs.")
    p.add_argument(
        "--mae-threshold",
        type=float,
        default=0.01,
        help="Report pass/fail vs course requirement (mean abs error).",
    )
    p.add_argument(
        "--exit-on-mae-fail",
        action="store_true",
        help="Exit with code 1 if final MAE >= --mae-threshold (for CI / batch jobs).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    layer_sizes = _parse_layers(args.layers)
    if len(layer_sizes) < 2 or layer_sizes[0] != 1 or layer_sizes[-1] != 1:
        print(
            "Warning: sin regression expects input dim 1 and output dim 1; "
            f"got {layer_sizes}.",
            file=sys.stderr,
        )

    grad_clip = None if args.grad_clip < 0 else args.grad_clip

    backend.init_backend(use_cuda=not args.no_cuda, gpu_id=args.gpu)
    xp = backend.get_xp()
    set_random_seed(args.seed)

    net = MLP(list(layer_sizes), hidden_activation=args.activation)
    criterion = MSELoss()
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

    batches = max(1, args.samples_per_epoch // args.batch_size)
    mode = f"CuPy GPU {backend.device_id()}" if backend.is_cuda() else "NumPy CPU"
    print(f"Backend: {mode} | layers={layer_sizes} | batches/epoch={batches}")

    for epoch in range(args.epochs):
        opt.lr = lr_fn(epoch)
        epoch_loss = 0.0
        for _ in range(batches):
            x = xp.random.uniform(
                -math.pi, math.pi, size=(args.batch_size, 1)
            ).astype(xp.float32)
            y = xp.sin(x)
            pred = net.forward(x)
            loss = criterion.forward(pred, y)
            epoch_loss += float(backend.to_cpu_array(loss))
            g = criterion.backward()
            clip_tensor_l2_norm(g, grad_clip)
            net.backward(g)
            opt.step()
        epoch_loss /= batches

        if epoch % args.log_every == 0 or epoch == args.epochs - 1:
            mae = eval_grid_mae(lambda t: net.forward(t), xp, n_points=10_000)
            ok = "OK" if mae < args.mae_threshold else "need tuning"
            print(
                f"epoch {epoch:5d}  lr={opt.lr:.6g}  "
                f"train_mse≈{epoch_loss:.6f}  eval_mae={mae:.6f}  ({ok} vs {args.mae_threshold})"
            )

    final_mae = eval_grid_mae(lambda t: net.forward(t), xp, n_points=50_000)
    print(f"Final eval MAE (50k samples): {final_mae:.6f}")
    if args.exit_on_mae_fail and final_mae >= args.mae_threshold:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
