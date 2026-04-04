"""
Compute backend: NumPy (CPU) or CuPy (CUDA).

Design goals
------------
- Single code path for training logic: all modules import ``xp`` and use NumPy/CuPy
  compatible calls only.
- Explicit device selection for multi-GPU servers (e.g. physical GPUs 4--7).
- Fail fast with a clear message if GPU is requested but CuPy is missing.

Note on course constraints
--------------------------
CuPy provides ndarray + elementwise/reduction ops like NumPy; it does **not**
provide autograd or neural-network modules, so it is not a "deep learning
framework" in the sense forbidden by the project statement.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

# Type alias for the active array module (numpy or cupy).
ArrayModule = type(__import__("numpy"))

_xp: ArrayModule | None = None
_using_cuda: bool = False
_device_id: int | None = None


def is_cuda() -> bool:
    """Return True if the active backend is CuPy on a CUDA device."""
    return _using_cuda


def device_id() -> int | None:
    """Current CUDA device ordinal, or None if CPU."""
    return _device_id


def get_xp() -> ArrayModule:
    """
    Return the active array module (``numpy`` or ``cupy``).

    Raises
    ------
    RuntimeError
        If :func:`init_backend` has not been called yet.
    """
    if _xp is None:
        raise RuntimeError("backend not initialized; call init_backend() first")
    return _xp


def init_backend(*, use_cuda: bool = False, gpu_id: int | None = None) -> ArrayModule:
    """
    Initialize global array backend.

    Parameters
    ----------
    use_cuda
        If True, attempt to use CuPy on CUDA device ``gpu_id``.
    gpu_id
        Physical CUDA device index (0, 1, ..., or 4--7 on an 8-GPU node).
        Ignored when ``use_cuda`` is False.

    Returns
    -------
    module
        Either ``numpy`` or ``cupy`` as the unified ``xp`` API.

    Notes
    -----
    On shared clusters, prefer either:

    - ``CUDA_VISIBLE_DEVICES=4 python train_*.py`` (remaps to logical device 0), or
    - ``python train_*.py --gpu 4`` with this module selecting device 4 explicitly.

    Do not mix both unless you understand ordinal remapping.
    """
    global _xp, _using_cuda, _device_id

    if not use_cuda:
        import numpy as np

        _xp = np
        _using_cuda = False
        _device_id = None
        return _xp

    try:
        import cupy as cp  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "use_cuda=True but CuPy is not installed. "
            "Install a matching wheel, e.g. `pip install cupy-cuda12x`, "
            "or run with --no-cuda."
        ) from e

    if gpu_id is None:
        # Respect CUDA_VISIBLE_DEVICES: CuPy's device 0 is the first *visible* device.
        gpu_id = int(os.environ.get("CUPY_DEVICE", "0"))

    cp.cuda.Device(gpu_id).use()
    _xp = cp
    _using_cuda = True
    _device_id = gpu_id
    return _xp


@contextmanager
def device_scope(gpu_id: int) -> Iterator[None]:
    """
    Temporarily switch the current CUDA device (CuPy only).

    Useful for advanced multi-GPU data-parallel experiments. Part-1 training
    scripts use a single device by default.
    """
    xp = get_xp()
    if not is_cuda():
        yield
        return
    import cupy as cp  # type: ignore

    with cp.cuda.Device(gpu_id):
        yield


def to_cpu_array(arr):
    """Copy any backend array to a host ``numpy.ndarray`` (for logging/saving)."""
    xp = get_xp()
    if is_cuda():
        return xp.asnumpy(arr)
    return xp.asarray(arr)


def ensure_numpy(x):
    """Convert backend array to NumPy (no copy if already CPU NumPy)."""
    if hasattr(x, "get"):  # cupy
        return x.get()
    return x
