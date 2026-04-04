"""
Synthetic dataset for regression: ``y = sin(x)`` with ``x ∈ [-π, π]``.

Training scripts draw i.i.d. ``x`` each step (online sampling). This module
provides helpers for evaluation grids and optional fixed batches on CPU for
export/plotting.
"""

from __future__ import annotations

import math

from . import backend


def eval_grid_mae(model_forward, xp, *, n_points: int = 10_000) -> float:
    """
    Mean absolute error on ``n_points`` uniform samples in ``[-π, π]``.

    Parameters
    ----------
    model_forward
        Callable ``x_batch -> pred`` where ``x_batch`` is backend array (N, 1).
    xp
        Active array module (``numpy`` or ``cupy``).

    Returns
    -------
    float
        MAE on host (Python float), suitable for logging / pass-fail vs 0.01.
    """
    x = xp.random.uniform(-math.pi, math.pi, size=(n_points, 1)).astype(xp.float32)
    y_true = xp.sin(x)
    pred = model_forward(x)
    mae = xp.mean(xp.abs(pred - y_true))
    return float(backend.to_cpu_array(mae))
