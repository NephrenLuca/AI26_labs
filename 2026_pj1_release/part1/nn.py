"""Fully connected neural network with manual backprop, BN, Dropout, and GPU support."""

from __future__ import annotations

from typing import List, Sequence

import numpy as np


def init_backend(use_cuda: bool, gpu_id: int = 0, seed: int = 42):
    """Return (xp, to_cpu, rng) where xp is numpy or cupy."""
    if not use_cuda:
        rng = np.random.default_rng(seed)
        return np, (lambda x: x), rng
    try:
        import cupy as cp  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError("CuPy is not installed. Install a CUDA-matched cupy package.") from exc

    cp.cuda.Device(gpu_id).use()
    rng = cp.random.RandomState(seed)
    return cp, cp.asnumpy, rng


class NeuralNetwork:
    def __init__(
        self,
        layer_sizes: Sequence[int],
        hidden_activation: str = "relu",
        output_activation: str = "linear",
        seed: int = 42,
        xp=np,
        rng=None,
        use_batchnorm: bool = False,
        bn_momentum: float = 0.9,
        bn_eps: float = 1e-5,
        dropout: float = 0.0,
        optimizer: str = "adam",
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_eps: float = 1e-8,
    ) -> None:
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes must contain at least input and output dims.")
        if hidden_activation not in {"relu", "tanh", "sigmoid"}:
            raise ValueError("hidden_activation must be one of: relu/tanh/sigmoid")
        if output_activation not in {"linear", "tanh", "sigmoid"}:
            raise ValueError("output_activation must be one of: linear/tanh/sigmoid")
        if not (0.0 <= dropout < 1.0):
            raise ValueError("dropout must be in [0, 1).")
        if optimizer not in {"sgd", "adam"}:
            raise ValueError("optimizer must be 'sgd' or 'adam'.")

        self.xp = xp
        self.rng = rng if rng is not None else np.random.default_rng(seed)
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.use_batchnorm = use_batchnorm
        self.bn_momentum = float(bn_momentum)
        self.bn_eps = float(bn_eps)
        self.dropout = float(dropout)
        self.optimizer = optimizer
        self.adam_beta1 = float(adam_beta1)
        self.adam_beta2 = float(adam_beta2)
        self.adam_eps = float(adam_eps)
        self._opt_step = 0

        np_rng = np.random.default_rng(seed)
        self.weights: List = []
        self.biases: List = []
        for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
            limit = float(np.sqrt(6.0 / (in_dim + out_dim)))
            w_np = np_rng.uniform(-limit, limit, size=(in_dim, out_dim)).astype(np.float32)
            b_np = np.zeros((1, out_dim), dtype=np.float32)
            self.weights.append(self.xp.asarray(w_np))
            self.biases.append(self.xp.asarray(b_np))

        self.num_layers = len(self.weights)
        self.num_hidden = self.num_layers - 1
        self.bn_gamma: List = []
        self.bn_beta: List = []
        self.bn_running_mean: List = []
        self.bn_running_var: List = []
        for out_dim in layer_sizes[1:-1]:
            self.bn_gamma.append(self.xp.ones((1, out_dim), dtype=self.xp.float32))
            self.bn_beta.append(self.xp.zeros((1, out_dim), dtype=self.xp.float32))
            self.bn_running_mean.append(self.xp.zeros((1, out_dim), dtype=self.xp.float32))
            self.bn_running_var.append(self.xp.ones((1, out_dim), dtype=self.xp.float32))

        # Adam states for all trainable parameters.
        self._m_w = [self.xp.zeros_like(w) for w in self.weights]
        self._v_w = [self.xp.zeros_like(w) for w in self.weights]
        self._m_b = [self.xp.zeros_like(b) for b in self.biases]
        self._v_b = [self.xp.zeros_like(b) for b in self.biases]
        self._m_bg = [self.xp.zeros_like(g) for g in self.bn_gamma]
        self._v_bg = [self.xp.zeros_like(g) for g in self.bn_gamma]
        self._m_bb = [self.xp.zeros_like(b) for b in self.bn_beta]
        self._v_bb = [self.xp.zeros_like(b) for b in self.bn_beta]

        self._cache_a: List = []
        self._cache_pre_act: List = []
        self._cache_dropout_masks: List = []
        self._cache_bn: List = []

    def _random_like(self, shape):
        """Sample U[0,1) with Generator- or RandomState-like RNGs."""
        if hasattr(self.rng, "random"):
            return self.rng.random(shape)
        if hasattr(self.rng, "rand"):
            return self.rng.rand(*shape)
        return self.xp.random.random(shape)

    def _activate(self, x, kind: str):
        if kind == "relu":
            return self.xp.maximum(x, 0.0)
        if kind == "tanh":
            return self.xp.tanh(x)
        if kind == "sigmoid":
            return 1.0 / (1.0 + self.xp.exp(-x))
        return x

    def _activation_grad(self, x, kind: str):
        if kind == "relu":
            return (x > 0).astype(x.dtype)
        if kind == "tanh":
            y = self.xp.tanh(x)
            return 1.0 - y * y
        if kind == "sigmoid":
            y = 1.0 / (1.0 + self.xp.exp(-x))
            return y * (1.0 - y)
        return self.xp.ones_like(x)

    def _batchnorm_forward(self, z, hidden_idx: int, training: bool):
        if training:
            mu = self.xp.mean(z, axis=0, keepdims=True)
            var = self.xp.var(z, axis=0, keepdims=True)
            z_centered = z - mu
            std_inv = 1.0 / self.xp.sqrt(var + self.bn_eps)
            z_norm = z_centered * std_inv
            out = self.bn_gamma[hidden_idx] * z_norm + self.bn_beta[hidden_idx]
            self.bn_running_mean[hidden_idx] = (
                self.bn_momentum * self.bn_running_mean[hidden_idx] + (1.0 - self.bn_momentum) * mu
            )
            self.bn_running_var[hidden_idx] = (
                self.bn_momentum * self.bn_running_var[hidden_idx] + (1.0 - self.bn_momentum) * var
            )
            self._cache_bn[hidden_idx] = (z_norm, z_centered, std_inv, z.shape[0])
            return out
        z_norm = (z - self.bn_running_mean[hidden_idx]) / self.xp.sqrt(self.bn_running_var[hidden_idx] + self.bn_eps)
        return self.bn_gamma[hidden_idx] * z_norm + self.bn_beta[hidden_idx]

    def _batchnorm_backward(self, grad_out, hidden_idx: int):
        z_norm, z_centered, std_inv, batch_size = self._cache_bn[hidden_idx]
        gamma = self.bn_gamma[hidden_idx]

        dgamma = self.xp.sum(grad_out * z_norm, axis=0, keepdims=True)
        dbeta = self.xp.sum(grad_out, axis=0, keepdims=True)
        dz_norm = grad_out * gamma
        dvar = self.xp.sum(dz_norm * z_centered * (-0.5) * (std_inv**3), axis=0, keepdims=True)
        dmu = self.xp.sum(dz_norm * (-std_inv), axis=0, keepdims=True) + dvar * self.xp.mean(
            -2.0 * z_centered, axis=0, keepdims=True
        )
        dz = dz_norm * std_inv + dvar * (2.0 / batch_size) * z_centered + dmu / batch_size
        return dz, dgamma, dbeta

    def _apply_update(self, param, grad, m, v, lr: float):
        if self.optimizer == "sgd":
            param -= lr * grad
            return
        m *= self.adam_beta1
        m += (1.0 - self.adam_beta1) * grad
        v *= self.adam_beta2
        v += (1.0 - self.adam_beta2) * (grad * grad)
        bias_c1 = 1.0 - (self.adam_beta1**self._opt_step)
        bias_c2 = 1.0 - (self.adam_beta2**self._opt_step)
        m_hat = m / bias_c1
        v_hat = v / bias_c2
        param -= lr * m_hat / (self.xp.sqrt(v_hat) + self.adam_eps)

    def forward(self, x, training: bool = True):
        a = self.xp.asarray(x, dtype=self.xp.float32)
        self._cache_a = [a]
        self._cache_pre_act = []
        self._cache_dropout_masks = [None for _ in range(self.num_hidden)]
        self._cache_bn = [None for _ in range(self.num_hidden)]

        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z_linear = a @ w + b
            is_last = i == self.num_layers - 1
            if is_last:
                pre_act = z_linear
                a = self._activate(pre_act, self.output_activation)
            else:
                pre_act = z_linear
                if self.use_batchnorm:
                    pre_act = self._batchnorm_forward(pre_act, i, training=training)
                a = self._activate(pre_act, self.hidden_activation)
                if training and self.dropout > 0.0:
                    mask = (self._random_like(a.shape) >= self.dropout).astype(a.dtype) / (1.0 - self.dropout)
                    a = a * mask
                    self._cache_dropout_masks[i] = mask
            self._cache_pre_act.append(pre_act)
            self._cache_a.append(a)
        return a

    def backward(self, grad_output, lr: float) -> None:
        if not self._cache_a or not self._cache_pre_act:
            raise RuntimeError("forward must be called before backward.")

        self._opt_step += 1
        grad = self.xp.asarray(grad_output, dtype=self.xp.float32)
        for i in reversed(range(self.num_layers)):
            pre_act = self._cache_pre_act[i]
            a_prev = self._cache_a[i]
            is_last = i == self.num_layers - 1

            if is_last:
                grad_pre = grad * self._activation_grad(pre_act, self.output_activation)
                grad_linear = grad_pre
            else:
                if self.dropout > 0.0 and self._cache_dropout_masks[i] is not None:
                    grad = grad * self._cache_dropout_masks[i]
                grad_pre = grad * self._activation_grad(pre_act, self.hidden_activation)
                if self.use_batchnorm:
                    grad_linear, dgamma, dbeta = self._batchnorm_backward(grad_pre, i)
                    self._apply_update(self.bn_gamma[i], dgamma, self._m_bg[i], self._v_bg[i], lr)
                    self._apply_update(self.bn_beta[i], dbeta, self._m_bb[i], self._v_bb[i], lr)
                else:
                    grad_linear = grad_pre

            batch_size = a_prev.shape[0]
            grad_w = (a_prev.T @ grad_linear) / batch_size
            grad_b = self.xp.mean(grad_linear, axis=0, keepdims=True)
            grad = grad_linear @ self.weights[i].T
            self._apply_update(self.weights[i], grad_w, self._m_w[i], self._v_w[i], lr)
            self._apply_update(self.biases[i], grad_b, self._m_b[i], self._v_b[i], lr)
