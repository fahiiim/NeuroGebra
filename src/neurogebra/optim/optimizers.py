"""
First-class Optimizers for NeuroGebra.

Standalone optimizer instances that manage parameter states and updates.
"""

from __future__ import annotations

from typing import Any
import numpy as np


class Optimizer:
    """Base class for all NeuroGebra optimizers."""

    def __init__(self, lr: float = 0.001):
        self.lr = float(lr)
        self.learning_rate = self.lr
        self.state: dict[str, Any] = {}
        self.t = 0

    def zero_grad(self, params: list[Any]) -> None:
        """Zero out gradients on all parameter objects."""
        for p in params:
            if hasattr(p, "grad") and p.grad is not None:
                if isinstance(p.grad, np.ndarray):
                    p.grad.fill(0)
                else:
                    p.grad = 0.0

    def step(self, params: list[Any]) -> None:
        """Perform a single optimization step."""
        raise NotImplementedError


class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer with optional momentum and Nesterov.

    Args:
        lr: Learning rate.
        momentum: Momentum factor (default: 0.0).
        nesterov: Enables Nesterov momentum (default: False).
        weight_decay: Weight decay (L2 penalty) factor (default: 0.0).
    """

    def __init__(
        self,
        lr: float = 0.01,
        momentum: float = 0.0,
        nesterov: bool = False,
        weight_decay: float = 0.0,
    ):
        super().__init__(lr=lr)
        self.momentum = momentum
        self.nesterov = nesterov
        self.weight_decay = weight_decay
        self.velocities: list[np.ndarray | None] = []

    def step(self, params: list[Any]) -> None:
        self.t += 1
        if not self.velocities:
            self.velocities = [
                np.zeros_like(p.data) if hasattr(p, "data") else None for p in params
            ]

        for i, p in enumerate(params):
            if not hasattr(p, "grad") or p.grad is None:
                continue

            grad = np.asarray(p.grad, dtype=np.float64)
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * p.data

            if self.momentum != 0:
                if self.velocities[i] is None:
                    self.velocities[i] = np.zeros_like(p.data)
                self.velocities[i] = self.momentum * self.velocities[i] + grad
                if self.nesterov:
                    step_val = self.momentum * self.velocities[i] + grad
                else:
                    step_val = self.velocities[i]
                p.data -= self.learning_rate * step_val
            else:
                p.data -= self.learning_rate * grad


class Adam(Optimizer):
    """
    Adam (Adaptive Moment Estimation) optimizer.

    Args:
        lr: Learning rate (default: 0.001).
        betas: Coefficients used for computing running averages (default: (0.9, 0.999)).
        eps: Term added to the denominator for numerical stability (default: 1e-8).
        weight_decay: Weight decay (L2 penalty) factor (default: 0.0).
    """

    def __init__(
        self,
        lr: float = 0.001,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        super().__init__(lr=lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m: list[np.ndarray | None] = []
        self.v: list[np.ndarray | None] = []

    def step(self, params: list[Any]) -> None:
        self.t += 1
        if not self.m:
            self.m = [
                np.zeros_like(p.data) if hasattr(p, "data") else None for p in params
            ]
            self.v = [
                np.zeros_like(p.data) if hasattr(p, "data") else None for p in params
            ]

        lr = self.learning_rate
        b1, b2 = self.beta1, self.beta2
        eps = self.eps
        bias_correction1 = 1.0 - b1**self.t
        bias_correction2 = 1.0 - b2**self.t

        for i, p in enumerate(params):
            if not hasattr(p, "grad") or p.grad is None:
                continue

            grad = np.asarray(p.grad, dtype=np.float64)
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * p.data

            if self.m[i] is None:
                self.m[i] = np.zeros_like(p.data)
                self.v[i] = np.zeros_like(p.data)

            self.m[i] = b1 * self.m[i] + (1 - b1) * grad
            self.v[i] = b2 * self.v[i] + (1 - b2) * (grad**2)

            m_hat = self.m[i] / bias_correction1
            v_hat = self.v[i] / bias_correction2

            p.data -= lr * m_hat / (np.sqrt(v_hat) + eps)


class RMSprop(Optimizer):
    """
    RMSprop optimizer.

    Args:
        lr: Learning rate (default: 0.01).
        alpha: Smoothing constant (default: 0.99).
        eps: Term added to the denominator for numerical stability (default: 1e-8).
        weight_decay: Weight decay (L2 penalty) factor (default: 0.0).
    """

    def __init__(
        self,
        lr: float = 0.01,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        super().__init__(lr=lr)
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        self.square_avgs: list[np.ndarray | None] = []

    def step(self, params: list[Any]) -> None:
        self.t += 1
        if not self.square_avgs:
            self.square_avgs = [
                np.zeros_like(p.data) if hasattr(p, "data") else None for p in params
            ]

        for i, p in enumerate(params):
            if not hasattr(p, "grad") or p.grad is None:
                continue

            grad = np.asarray(p.grad, dtype=np.float64)
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * p.data

            if self.square_avgs[i] is None:
                self.square_avgs[i] = np.zeros_like(p.data)

            self.square_avgs[i] = self.alpha * self.square_avgs[i] + (
                1.0 - self.alpha
            ) * (grad**2)
            avg = np.sqrt(self.square_avgs[i]) + self.eps
            p.data -= self.learning_rate * grad / avg
