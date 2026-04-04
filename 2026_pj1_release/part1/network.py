"""
Composable MLP with manual backpropagation.

The network is a list of alternating linear transforms and element-wise
nonlinearities. The last layer outputs raw logits (classification) or
scalar/vector targets (regression).

Backward pass walks the layer list in reverse, chaining Jacobian-vector
products implemented per module.
"""

from __future__ import annotations

from typing import Protocol

from . import backend
from .layers import Linear, ReLU, Tanh


class _Module(Protocol):
    def forward(self, x): ...
    def backward(self, grad_out): ...


class MLP:
    """
    Multi-layer perceptron built from :class:`~part1.layers.Linear` and activations.

    Parameters
    ----------
    layer_sizes
        e.g. ``[1, 64, 64, 1]`` for scalar sin regression.
    hidden_activation
        ``"tanh"`` (smooth; good for regression) or ``"relu"``.
    """

    def __init__(
        self,
        layer_sizes: list[int],
        *,
        hidden_activation: str = "tanh",
    ) -> None:
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes must have at least input and output dim")

        self.layers: list[_Module] = []
        act_ctor = Tanh if hidden_activation == "tanh" else ReLU

        for i in range(len(layer_sizes) - 1):
            in_f, out_f = layer_sizes[i], layer_sizes[i + 1]
            is_hidden = i < len(layer_sizes) - 2
            # Hidden layers: slight negative bias (course hint); output layer: zero.
            b0 = -0.1 if is_hidden else 0.0
            self.layers.append(Linear(in_f, out_f, bias_init=b0))
            if is_hidden:
                self.layers.append(act_ctor())

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad_out):
        for layer in reversed(self.layers):
            grad_out = layer.backward(grad_out)
        return grad_out

    def linear_layers(self) -> list[Linear]:
        """Return all :class:`~part1.layers.Linear` modules in forward order."""
        return [layer for layer in self.layers if isinstance(layer, Linear)]
