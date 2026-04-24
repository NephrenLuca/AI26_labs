"""Array-library backend abstraction.

Reads ``UNPYTORCHED_DEVICE`` (``"cpu"`` or ``"gpu"``) at import time and
binds ``xp`` to :mod:`numpy` or :mod:`cupy` accordingly. Downstream modules
(``mynn``, ``train``, ``predict``) should import ``xp`` from here instead of
importing ``cupy`` / ``numpy`` directly.

The environment variable must be set **before** this module is imported.
Entry-point scripts parse ``--device`` first and then set the env var.
"""

from __future__ import annotations

import os
from typing import Any

_DEVICE = os.environ.get("UNPYTORCHED_DEVICE", "gpu").lower()

if _DEVICE not in {"cpu", "gpu"}:
    raise ValueError(
        f"UNPYTORCHED_DEVICE must be 'cpu' or 'gpu', got {_DEVICE!r}"
    )

if _DEVICE == "cpu":
    import numpy as xp  # type: ignore

    is_gpu: bool = False
    device: str = "cpu"
else:
    try:
        import cupy as xp  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "cupy is not installed but UNPYTORCHED_DEVICE=gpu was requested. "
            "Install cupy (e.g. `pip install cupy-cuda12x`) or rerun with "
            "--device cpu."
        ) from exc

    is_gpu = True
    device = "gpu"


def to_numpy(a: Any):
    """Return a :class:`numpy.ndarray` view of ``a`` regardless of backend."""
    if is_gpu:
        return xp.asnumpy(a)
    return a


def set_gpu_device(gpu_id: int) -> None:
    """Select the active CUDA device. No-op on CPU backend."""
    if is_gpu:
        xp.cuda.Device(gpu_id).use()


def seed(value: int) -> None:
    """Seed the active backend's RNG."""
    xp.random.seed(value)


__all__ = ["xp", "is_gpu", "device", "to_numpy", "set_gpu_device", "seed"]
