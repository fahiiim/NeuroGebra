"""
Learning Rate Schedulers for NeuroGebra.

Provides dynamic learning rate scheduling algorithms compatible with
NeuroGebra ModelBuilder models and optimizers.
"""

from __future__ import annotations

import math
from typing import Any


class LRScheduler:
    """
    Base class for all learning rate schedulers.

    Args:
        target: A Model, ModelBuilder, or Optimizer instance having a
            `learning_rate` or `lr` attribute.
        last_epoch: The index of the last epoch. Default: -1.
    """

    def __init__(self, target: Any, last_epoch: int = -1):
        self.target = target
        if hasattr(target, "learning_rate"):
            self.base_lr = float(target.learning_rate)
        elif hasattr(target, "lr"):
            self.base_lr = float(target.lr)
        else:
            self.base_lr = 0.001

        self.last_epoch = last_epoch
        self.current_lr = self.base_lr
        self.step()

    def get_lr(self) -> float:
        """Compute the learning rate for the current epoch."""
        raise NotImplementedError

    def step(self, epoch: int | None = None) -> float:
        """
        Step the scheduler forward by one epoch or to a specific epoch.

        Args:
            epoch: Optional target epoch index.

        Returns:
            The newly updated learning rate.
        """
        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = epoch

        self.current_lr = float(self.get_lr())

        if hasattr(self.target, "learning_rate"):
            self.target.learning_rate = self.current_lr
        elif hasattr(self.target, "lr"):
            self.target.lr = self.current_lr

        return self.current_lr


class StepLR(LRScheduler):
    """
    Decays the learning rate by gamma every step_size epochs.

    Formula:
        lr = base_lr * (gamma ** (epoch // step_size))
    """

    def __init__(
        self,
        target: Any,
        step_size: int,
        gamma: float = 0.1,
        last_epoch: int = -1,
    ):
        if step_size <= 0:
            raise ValueError(f"step_size must be positive, got {step_size}")
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(target, last_epoch)

    def get_lr(self) -> float:
        if self.last_epoch == 0:
            return self.base_lr
        return self.base_lr * (self.gamma ** (self.last_epoch // self.step_size))


class CosineAnnealingLR(LRScheduler):
    """
    Cosine annealing learning rate schedule.

    Formula:
        lr = eta_min + 0.5 * (base_lr - eta_min) * (1 + cos(pi * epoch / T_max))
    """

    def __init__(
        self,
        target: Any,
        T_max: int,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ):
        if T_max <= 0:
            raise ValueError(f"T_max must be positive, got {T_max}")
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(target, last_epoch)

    def get_lr(self) -> float:
        if self.last_epoch == 0:
            return self.base_lr
        fraction = min(1.0, self.last_epoch / self.T_max)
        return self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (
            1.0 + math.cos(math.pi * fraction)
        )


class ExponentialLR(LRScheduler):
    """
    Decays the learning rate by gamma every epoch.

    Formula:
        lr = base_lr * (gamma ** epoch)
    """

    def __init__(
        self,
        target: Any,
        gamma: float = 0.95,
        last_epoch: int = -1,
    ):
        if gamma <= 0:
            raise ValueError(f"gamma must be positive, got {gamma}")
        self.gamma = gamma
        super().__init__(target, last_epoch)

    def get_lr(self) -> float:
        return self.base_lr * (self.gamma**self.last_epoch)


class ReduceLROnPlateau:
    """
    Reduce learning rate when a metric has stopped improving.

    Args:
        target: Model or Optimizer having learning_rate or lr.
        mode: 'min' or 'max'.
        factor: Factor by which the learning rate will be reduced. lr = lr * factor.
        patience: Number of epochs with no improvement after which lr is reduced.
        min_lr: Lower bound on the learning rate.
    """

    def __init__(
        self,
        target: Any,
        mode: str = "min",
        factor: float = 0.5,
        patience: int = 5,
        min_lr: float = 1e-6,
    ):
        self.target = target
        if hasattr(target, "learning_rate"):
            self.current_lr = float(target.learning_rate)
        elif hasattr(target, "lr"):
            self.current_lr = float(target.lr)
        else:
            self.current_lr = 0.001

        self.mode = mode.lower()
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.best = float("inf") if self.mode == "min" else float("-inf")
        self.num_bad_epochs = 0

    def step(self, metric: float) -> float:
        """
        Update the learning rate based on current epoch metric.

        Args:
            metric: Loss or performance metric value to evaluate.

        Returns:
            The updated learning rate.
        """
        is_better = (
            (metric < self.best) if self.mode == "min" else (metric > self.best)
        )
        if is_better:
            self.best = metric
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs > self.patience:
            self.current_lr = max(self.min_lr, self.current_lr * self.factor)
            if hasattr(self.target, "learning_rate"):
                self.target.learning_rate = self.current_lr
            elif hasattr(self.target, "lr"):
                self.target.lr = self.current_lr
            self.num_bad_epochs = 0

        return self.current_lr
