"""
Shared training utilities: reproducibility and gradient clipping.

Gradient clipping limits the L2 norm of the tensor returned by the loss
``backward()`` (i.e. dL/dlogits or dL/dpred) before it is propagated through
the MLP. This dampens rare exploding steps without changing the rest of BP.
"""

from __future__ import annotations

import random

from . import backend


def set_random_seed(seed: int) -> None:
    """
    Seed Python, NumPy, and CuPy RNGs (when CUDA is active).

    Call **after** :func:`~part1.backend.init_backend` so the correct ``xp``
    backend is configured.
    """
    random.seed(seed)
    xp = backend.get_xp()
    if backend.is_cuda():
        # CuPy maintains its own generator; NumPy seed does not affect GPU.
        xp.random.seed(seed)
    else:
        xp.random.seed(seed)


def clip_tensor_l2_norm(grad, max_norm: float | None):
    """
    Scale ``grad`` in-place if its Frobenius norm exceeds ``max_norm``.

    Returns
    -------
    grad
        Same array object, possibly scaled.
    """
    if max_norm is None:
        return grad
    xp = backend.get_xp()
    gn = xp.sqrt(xp.sum(grad * grad, dtype=xp.float64))
    gn_f = float(backend.to_cpu_array(gn))
    if gn_f > max_norm and gn_f > 0.0:
        grad *= max_norm / gn_f
    return grad
