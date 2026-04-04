"""
Optimizers updating learnable tensors (here: Linear weights and biases).

Part 1 uses vanilla SGD with optional momentum and optional L2 weight decay
(decoupled weight decay, applied to weights only).

Learning-rate schedules (step decay, cosine) are provided as small callables
to keep training scripts readable.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from . import backend

if TYPE_CHECKING:
    from .layers import Linear


class SGDOptimizer:
    """
    Stochastic gradient descent with momentum.

    For each linear layer, after :meth:`~part1.layers.Linear.backward`::

        vW = mu * vW - lr * dW
        vb = mu * vb - lr * db
        W += vW; b += vb

    Weight decay (L2), when enabled::

        W -= lr * wd * W
    """

    def __init__(
        self,
        linears: list[Linear],
        lr: float,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ) -> None:
        self.linears = linears
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self._velocities: dict[int, dict[str, object]] = {}

    def step(self) -> None:
        xp = backend.get_xp()
        for lin in self.linears:
            pid = id(lin.W)
            if pid not in self._velocities:
                self._velocities[pid] = {
                    "W": xp.zeros_like(lin.W),
                    "b": xp.zeros_like(lin.b),
                }
            v = self._velocities[pid]

            v["W"] = self.momentum * v["W"] - self.lr * lin.dW
            v["b"] = self.momentum * v["b"] - self.lr * lin.db

            lin.W += v["W"]
            lin.b += v["b"]

            if self.weight_decay > 0.0:
                lin.W -= self.lr * self.weight_decay * lin.W


def lr_step(epoch: int, base_lr: float, step_size: int, gamma: float) -> float:
    """Piecewise constant decay: multiply by ``gamma`` every ``step_size`` epochs."""
    return base_lr * (gamma ** (epoch // step_size))


def lr_cosine(epoch: int, total_epochs: int, base_lr: float, eta_min: float) -> float:
    """Cosine decay from ``base_lr`` to ``eta_min``."""
    import math

    if total_epochs <= 1:
        return base_lr
    t = epoch / max(total_epochs - 1, 1)
    return eta_min + 0.5 * (base_lr - eta_min) * (1.0 + math.cos(math.pi * t))


def make_lr_fn(
    schedule: str,
    base_lr: float,
    total_epochs: int,
    *,
    step_size: int = 500,
    gamma: float = 0.5,
    eta_min: float = 1e-5,
) -> Callable[[int], float]:
    """
    Factory for epoch-based learning-rate functions.

    Parameters
    ----------
    schedule
        ``"constant"``, ``"step"``, or ``"cosine"``.
    """
    if schedule == "constant":

        def f(_e: int) -> float:
            return base_lr

        return f
    if schedule == "step":

        def f(e: int) -> float:
            return lr_step(e, base_lr, step_size, gamma)

        return f
    if schedule == "cosine":

        def f(e: int) -> float:
            return lr_cosine(e, total_epochs, base_lr, eta_min)

        return f
    raise ValueError(f"unknown schedule: {schedule}")
