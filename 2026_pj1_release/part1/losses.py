"""
Loss functions and their derivatives for backpropagation.

Each loss implements:
- ``forward(pred, target) -> scalar loss`` (stored internally for logging)
- ``backward()`` -> gradient w.r.t. ``pred`` with same shape as ``pred``

Regression uses mean squared error (smooth gradients for sin fitting).
Classification uses softmax + cross-entropy (multi-class, numerically stable).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import backend

if TYPE_CHECKING:
    pass


class MSELoss:
    """
    Mean squared error averaged over batch and features.

    For vector targets shape (N, D): L = mean( (pred - target)^2 ).

    Backward
    --------
    dL/dpred = 2 * (pred - target) / (N * D)
    """

    def __init__(self) -> None:
        self._pred = None
        self._target = None
        self.last_loss: float | None = None

    def forward(self, pred, target):
        xp = backend.get_xp()
        self._pred = pred
        self._target = target
        diff = pred - target
        n = max(pred.size, 1)
        loss = xp.sum(diff * diff) / n
        self.last_loss = float(backend.to_cpu_array(loss))
        return loss

    def backward(self):
        xp = backend.get_xp()
        pred, target = self._pred, self._target
        n = max(pred.size, 1)
        return 2.0 * (pred - target) / n


class CrossEntropyLoss:
    """
    Softmax cross-entropy for one-hot or integer labels.

    Parameters
    ----------
    pred
        Logits of shape (N, C) — **no softmax applied yet**.
    target
        Either integer labels (N,) with values in [0, C-1], or one-hot (N, C).

    Forward applies log-softmax internally for numerical stability:
    L_i = -log( exp(z_yi) / sum_j exp(z_j) )

    Backward
    --------
    With softmax p = softmax(z), one-hot y: dL/dz = (p - y) / N  (mean reduction).
    """

    def __init__(self) -> None:
        self._logits = None
        self._target = None
        self._probs = None
        self._target_one_hot = None
        self.last_loss: float | None = None

    @staticmethod
    def _labels_to_one_hot(xp, labels, num_classes: int):
        n = int(labels.shape[0])
        oh = xp.zeros((n, num_classes), dtype=xp.float32)
        idx = labels.astype(xp.int32)
        rows = xp.arange(n, dtype=xp.int32)
        oh[rows, idx] = 1.0
        return oh

    def forward(self, logits, target):
        xp = backend.get_xp()
        self._logits = logits
        n, c = logits.shape

        if target.ndim == 1:
            self._target_one_hot = self._labels_to_one_hot(xp, target, c)
        else:
            self._target_one_hot = target.astype(logits.dtype)

        # Stable softmax: subtract row-wise max.
        m = xp.max(logits, axis=1, keepdims=True)
        ex = xp.exp(logits - m)
        s = xp.sum(ex, axis=1, keepdims=True)
        probs = ex / s
        self._probs = probs

        log_p = xp.log(probs + 1e-30)
        # Cross-entropy: -sum_k y_ik log p_ik, mean over batch
        ce = -xp.sum(self._target_one_hot * log_p, axis=1)
        loss = xp.mean(ce)
        self.last_loss = float(backend.to_cpu_array(loss))
        return loss

    def backward(self):
        xp = backend.get_xp()
        n = self._logits.shape[0]
        return (self._probs - self._target_one_hot) / n
