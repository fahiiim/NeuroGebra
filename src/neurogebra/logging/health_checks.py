"""
SmartHealthChecker — Intelligent training diagnostics.

Detects vanishing/exploding gradients, dead neurons, overfitting,
underfitting, NaN/Inf corruption, learning stagnation, and more.
Every alert includes a human-readable diagnosis, severity colour,
and 2-3 concrete, actionable recommendations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class HealthAlert:
    """A single health diagnostic result."""
    check_name: str
    severity: str            # "info" | "warning" | "danger" | "critical"
    message: str
    diagnosis: str
    recommendations: List[str] = field(default_factory=list)
    data: Dict[str, Any] = field(default_factory=dict)


class SmartHealthChecker:
    """
    Run a battery of health checks on training metrics, gradients,
    weights, and activations.  Returns structured ``HealthAlert`` objects
    that can be logged or displayed.

    Usage::

        checker = SmartHealthChecker()
        alerts = checker.run_all(
            epoch=5,
            train_losses=[1.2, 1.1, 1.05, 1.04, 1.039, 1.038],
            val_losses=[1.3, 1.4, 1.5, 1.6, 1.7, 1.8],
            gradient_norms={"dense_1": 0.001, "dense_2": 1e-9},
            weight_stats={"dense_1": {"zeros_pct": 70}},
            activation_stats={"dense_1": {"zeros_pct": 80, "activation_type": "relu"}},
        )
    """

    def __init__(
        self,
        patience: int = 5,
        overfit_ratio: float = 1.5,
        stagnation_eps: float = 1e-4,
        gradient_vanish_thresh: float = 1e-7,
        gradient_explode_thresh: float = 100.0,
        dead_neuron_pct: float = 50.0,
    ):
        self.patience = patience
        self.overfit_ratio = overfit_ratio
        self.stagnation_eps = stagnation_eps
        self.gradient_vanish_thresh = gradient_vanish_thresh
        self.gradient_explode_thresh = gradient_explode_thresh
        self.dead_neuron_pct = dead_neuron_pct

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run_all(
        self,
        epoch: int,
        train_losses: Optional[List[float]] = None,
        val_losses: Optional[List[float]] = None,
        train_accs: Optional[List[float]] = None,
        val_accs: Optional[List[float]] = None,
        gradient_norms: Optional[Dict[str, float]] = None,
        weight_stats: Optional[Dict[str, Dict]] = None,
        activation_stats: Optional[Dict[str, Dict]] = None,
        current_lr: Optional[float] = None,
    ) -> List[HealthAlert]:
        alerts: List[HealthAlert] = []

        # NaN / Inf check (always first)
        alerts.extend(self._check_nan_inf(train_losses, val_losses))

        if train_losses and val_losses:
            alerts.extend(self._check_overfitting(epoch, train_losses, val_losses))
            alerts.extend(self._check_underfitting(epoch, train_losses, val_losses))

        if train_losses:
            alerts.extend(self._check_stagnation(epoch, train_losses))
            alerts.extend(self._check_divergence(train_losses))

        if gradient_norms:
            alerts.extend(self._check_gradients(gradient_norms))

        if weight_stats:
            alerts.extend(self._check_weights(weight_stats))

        if activation_stats:
            alerts.extend(self._check_activations(activation_stats))

        # Success message
        if not alerts and epoch > 0:
            alerts.append(HealthAlert(
                check_name="all_clear",
                severity="success",
                message="All systems healthy",
                diagnosis="Training is progressing normally.",
                recommendations=[],
            ))

        return alerts

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_nan_inf(self, train_losses, val_losses) -> List[HealthAlert]:
        alerts = []
        for label, arr in [("train_loss", train_losses), ("val_loss", val_losses)]:
            if arr is None:
                continue
            if any(np.isnan(v) or np.isinf(v) for v in arr):
                alerts.append(HealthAlert(
                    check_name="nan_inf",
                    severity="critical",
                    message=f"NaN/Inf detected in {label}!",
                    diagnosis=(
                        "Numerical instability has corrupted the loss. "
                        "Training should be stopped immediately."
                    ),
                    recommendations=[
                        "Lower the learning rate (current may be too high)",
                        "Add gradient clipping (max_norm=1.0)",
                        "Check input data for NaN/Inf values",
                        "Use a more numerically stable loss (e.g., log-sum-exp trick)",
                    ],
                    data={"series": label, "values": list(arr[-5:])},
                ))
        return alerts

    def _check_overfitting(self, epoch, train_losses, val_losses) -> List[HealthAlert]:
        if epoch < self.patience:
            return []
        recent_train = float(np.mean(train_losses[-self.patience:]))
        recent_val = float(np.mean(val_losses[-self.patience:]))
        if recent_train < 1e-12:
            return []
        ratio = recent_val / max(recent_train, 1e-12)
        if ratio > self.overfit_ratio:
            return [HealthAlert(
                check_name="overfitting",
                severity="warning",
                message=f"Overfitting detected (val/train ratio = {ratio:.2f})",
                diagnosis=(
                    "Validation loss is significantly higher than training loss, "
                    "meaning the model is memorising training data instead of learning patterns."
                ),
                recommendations=[
                    "Add Dropout layers (rate=0.2-0.5)",
                    "Use L2 regularization (weight_decay=1e-4)",
                    "Get more training data or use data augmentation",
                    "Reduce model complexity (fewer layers / neurons)",
                ],
                data={"ratio": ratio, "recent_train": recent_train, "recent_val": recent_val},
            )]
        return []

    def _check_underfitting(self, epoch, train_losses, val_losses) -> List[HealthAlert]:
        if epoch < self.patience * 2:
            return []
        recent_train = float(np.mean(train_losses[-self.patience:]))
        recent_val = float(np.mean(val_losses[-self.patience:]))
        if recent_train > 1.0 and recent_val > 1.0:
            return [HealthAlert(
                check_name="underfitting",
                severity="warning",
                message="Model may be underfitting",
                diagnosis=(
                    "Both training and validation losses remain high, "
                    "indicating the model lacks capacity to learn the data."
                ),
                recommendations=[
                    "Increase model size (more layers or neurons)",
                    "Train for more epochs",
                    "Try a lower learning rate",
                    "Check that data is properly normalised",
                ],
                data={"train": recent_train, "val": recent_val},
            )]
        return []

    def _check_stagnation(self, epoch, train_losses) -> List[HealthAlert]:
        if len(train_losses) < self.patience * 2:
            return []
        recent = train_losses[-self.patience:]
        delta = abs(recent[-1] - recent[0])
        if delta < self.stagnation_eps:
            return [HealthAlert(
                check_name="stagnation",
                severity="warning",
                message=f"Loss stagnant for {self.patience} epochs (Δ={delta:.2e})",
                diagnosis="The loss has stopped improving, suggesting a learning plateau.",
                recommendations=[
                    "Reduce learning rate (try lr × 0.1)",
                    "Use learning rate scheduling (e.g., cosine annealing)",
                    "Try a different optimizer (switch SGD↔Adam)",
                ],
            )]
        return []

    def _check_divergence(self, train_losses) -> List[HealthAlert]:
        if len(train_losses) < 3:
            return []
        # Check if loss is increasing
        recent = train_losses[-3:]
        if recent[-1] > recent[0] * 2:
            return [HealthAlert(
                check_name="divergence",
                severity="danger",
                message="Loss is diverging!",
                diagnosis="The loss is increasing rapidly, indicating training instability.",
                recommendations=[
                    "Immediately lower the learning rate",
                    "Add gradient clipping",
                    "Check data preprocessing (normalise inputs)",
                ],
                data={"recent": recent},
            )]
        return []

    def _check_gradients(self, gradient_norms: Dict[str, float]) -> List[HealthAlert]:
        alerts = []
        for layer, norm in gradient_norms.items():
            if np.isnan(norm) or np.isinf(norm):
                alerts.append(HealthAlert(
                    check_name="gradient_nan",
                    severity="critical",
                    message=f"NaN/Inf gradient in layer '{layer}'",
                    diagnosis="Gradient corruption will prevent any further learning.",
                    recommendations=[
                        "Lower the learning rate",
                        "Add gradient clipping (max_norm=1.0)",
                        "Use batch normalisation before this layer",
                    ],
                ))
            elif norm < self.gradient_vanish_thresh:
                alerts.append(HealthAlert(
                    check_name="vanishing_gradient",
                    severity="danger",
                    message=f"Vanishing gradient in '{layer}' (norm={norm:.2e})",
                    diagnosis=(
                        "Gradients are too small for the weights to update, "
                        "so this layer is effectively frozen."
                    ),
                    recommendations=[
                        "Switch activation from sigmoid/tanh to ReLU or LeakyReLU",
                        "Use batch normalisation",
                        "Try skip connections (ResNet-style)",
                        "Use He or Xavier weight initialisation",
                    ],
                    data={"layer": layer, "norm": norm},
                ))
            elif norm > self.gradient_explode_thresh:
                alerts.append(HealthAlert(
                    check_name="exploding_gradient",
                    severity="danger",
                    message=f"Exploding gradient in '{layer}' (norm={norm:.2e})",
                    diagnosis="Gradients are excessively large, causing unstable weight updates.",
                    recommendations=[
                        "Add gradient clipping (max_norm=1.0)",
                        "Lower the learning rate",
                        "Use batch normalisation",
                    ],
                    data={"layer": layer, "norm": norm},
                ))
        return alerts

    def _check_weights(self, weight_stats: Dict[str, Dict]) -> List[HealthAlert]:
        alerts = []
        for layer, stats in weight_stats.items():
            zeros_pct = stats.get("zeros_pct", 0)
            if zeros_pct > self.dead_neuron_pct:
                alerts.append(HealthAlert(
                    check_name="dead_neurons",
                    severity="warning",
                    message=f"{zeros_pct:.1f}% dead neurons in '{layer}'",
                    diagnosis=(
                        "A large fraction of weights are near zero, "
                        "meaning those neurons contribute nothing to the output."
                    ),
                    recommendations=[
                        "Switch to LeakyReLU or ELU activation",
                        "Use a different weight initialisation",
                        "Lower the learning rate to prevent weights from dying",
                    ],
                    data={"layer": layer, "zeros_pct": zeros_pct},
                ))
        return alerts

    def _check_activations(self, activation_stats: Dict[str, Dict]) -> List[HealthAlert]:
        alerts = []
        for layer, stats in activation_stats.items():
            zeros_pct = stats.get("zeros_pct", 0)
            act_type = stats.get("activation_type", "")
            if act_type in ("relu",) and zeros_pct > self.dead_neuron_pct:
                alerts.append(HealthAlert(
                    check_name="dead_relu",
                    severity="warning",
                    message=f"{zeros_pct:.1f}% dead ReLU in '{layer}'",
                    diagnosis="Neurons producing zero outputs will receive zero gradients and never recover.",
                    recommendations=[
                        "Use LeakyReLU(negative_slope=0.01) instead of ReLU",
                        "Lower learning rate",
                        "Use He initialisation",
                    ],
                ))
            sat_pct = stats.get("saturation_pct", 0)
            if sat_pct > 40:
                alerts.append(HealthAlert(
                    check_name="saturation",
                    severity="warning",
                    message=f"{sat_pct:.1f}% activations saturated in '{layer}'",
                    diagnosis="Saturated activations produce near-zero gradients (vanishing gradient problem).",
                    recommendations=[
                        "Switch to ReLU or GELU activation",
                        "Normalise inputs to the layer",
                        "Use batch normalisation",
                    ],
                ))
        return alerts
