"""
Differentiable building blocks with explicit forward/backward.

Naming convention
-----------------
- ``forward(x, ...)`` returns the layer output and caches whatever is needed
  for ``backward(grad_out)``.
- ``backward(grad_out)`` returns ``grad_in`` with the same shape as the
  layer's input ``x``.

Weight initialization follows Glorot (Xavier) uniform for linear layers,
which tends to work well for tanh/ReLU MLPs at moderate depth.
"""

from __future__ import annotations

import math

from . import backend


def _glorot_uniform(fan_in: int, fan_out: int, xp):
    """Glorot uniform bound for linear layers."""
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    return xp.random.uniform(-limit, limit, size=(fan_in, fan_out)).astype(xp.float32)


class Linear:
    """
    Affine map: y = x @ W + b

    Shapes
    ------
    x: (N, in_features)
    W: (in_features, out_features)
    b: (out_features,)
    y: (N, out_features)

    Notes
    -----
    Course guidance: small **negative** bias init on hidden layers can improve
    early stability when using ReLU (keeps many units inactive at start).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias_init: float = 0.0,
    ) -> None:
        xp = backend.get_xp()
        self.W = _glorot_uniform(in_features, out_features, xp)
        self.b = xp.full(out_features, bias_init, dtype=xp.float32)
        self._x = None

    def forward(self, x):
        self._x = x
        return x @ self.W + self.b

    def backward(self, grad_out):
        """
        grad_out: (N, out)

        dL/dW = x^T @ grad_out
        dL/db = sum_n grad_out[n]
        dL/dx = grad_out @ W^T
        """
        x = self._x
        self.dW = x.T @ grad_out
        self.db = backend.get_xp().sum(grad_out, axis=0)
        return grad_out @ self.W.T


class Tanh:
    """Hyperbolic tangent activation, element-wise."""

    def __init__(self) -> None:
        self._y = None

    def forward(self, x):
        xp = backend.get_xp()
        self._y = xp.tanh(x)
        return self._y

    def backward(self, grad_out):
        y = self._y
        return grad_out * (1.0 - y * y)


class ReLU:
    """ReLU activation: max(0, x), element-wise."""

    def __init__(self) -> None:
        self._mask = None

    def forward(self, x):
        xp = backend.get_xp()
        self._mask = x > 0
        return x * self._mask

    def backward(self, grad_out):
        return grad_out * self._mask


class Sigmoid:
    """Sigmoid activation (optional for binary or bounded outputs)."""

    def __init__(self) -> None:
        self._y = None

    def forward(self, x):
        xp = backend.get_xp()
        # Stable sigmoid for large |x|
        self._y = 1.0 / (1.0 + xp.exp(-xp.clip(x, -50.0, 50.0)))
        return self._y

    def backward(self, grad_out):
        y = self._y
        return grad_out * y * (1.0 - y)
