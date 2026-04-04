"""
Recommended hyperparameters for Part 1 tasks.

These defaults are tuned for:
- **Regression**: smooth ``sin(x)`` on ``[-pi, pi]`` with MSE; tanh hidden units
  and moderate width reach **MAE < 0.01** quickly when trained on GPU.
- **Classification**: 12-class grayscale character MLP; assumes roughly
  **28×28** inputs (resize if your dataset differs).

You may override every value via CLI flags in the training scripts. Document
your own sweeps in the course PDF.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SinRegressionHParams:
    """
    Hyperparameters for ``y = sin(x)``, ``x ∈ [-π, π]``.

    Rationale
    ---------
    - **tanh** hidden activations match smooth target; avoid ReLU dead zones on
      negative inputs.
    - **Layer sizes** ``[1, 128, 128, 1]``: wide enough to approximate sin,
      shallow enough to train stably without heavy tuning.
    - **LR 0.02** with **cosine decay** works well for MSE on this scale; if
      loss oscillates, halve LR or enable momentum 0.9.
    - **Samples per epoch** large enough that each epoch sees diverse ``x``.
    """

    layer_sizes: tuple[int, ...] = (1, 128, 128, 1)
    hidden_activation: str = "tanh"
    epochs: int = 4000
    samples_per_epoch: int = 4096
    batch_size: int = 256
    lr: float = 0.02
    lr_schedule: str = "cosine"  # constant | step | cosine
    lr_step_size: int = 1000
    lr_gamma: float = 0.5
    lr_eta_min: float = 1e-4
    momentum: float = 0.0
    weight_decay: float = 0.0
    grad_clip: float | None = 5.0  # L2 clip on backprop signal; None to disable


@dataclass(frozen=True)
class CharClassificationHParams:
    """
    Hyperparameters for 12-way character classification (MLP on flattened pixels).

    Rationale
    ---------
    - **ReLU** + deeper/wider hidden stack is standard for vision-ish inputs.
    - **LR 0.01** with **step decay** after a plateau is simple and robust.
    - **Weight decay 1e-4** mild regularization when training set is small.
    - **batch_size 128** balances noise and GPU utilization.
    """

    hidden_sizes: tuple[int, ...] = (512, 256, 128)
    hidden_activation: str = "relu"
    epochs: int = 80
    batch_size: int = 128
    lr: float = 0.01
    lr_schedule: str = "step"
    lr_step_size: int = 30
    lr_gamma: float = 0.5
    lr_eta_min: float = 1e-5
    momentum: float = 0.9
    weight_decay: float = 1e-4
    grad_clip: float | None = 10.0
    # If None, infer from first image in dataset
    image_size: tuple[int, int] | None = (28, 28)
    num_classes: int = 12


SIN_DEFAULTS = SinRegressionHParams()
CHAR_DEFAULTS = CharClassificationHParams()
