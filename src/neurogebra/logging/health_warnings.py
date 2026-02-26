"""
AutoHealthWarnings — Automated threshold-based health warning system.

Turns the Training Observatory from a passive data dump into an **active
diagnostic tool**.  Built-in rules fire whenever key metrics cross
configurable thresholds, emitting structured ``HealthWarning`` objects
with human-readable explanations and remediation advice.

Rules implemented:
    * ``zeros_pct > 50%``   → "Possible dying ReLU in <layer>"
    * Gradient-norm sudden spike  → "Possible exploding gradient"
    * ``val_loss`` diverging from ``train_loss`` → "Possible overfitting"
    * Weight delta ≈0 for N consecutive batches → "Optimizer may have stagnated"
    * Loss increasing for N batches → "Training diverging"
    * NaN / Inf anywhere → immediate critical alert
    * Learning-rate too high heuristic
    * Activation saturation

Usage::

    from neurogebra.logging.health_warnings import AutoHealthWarnings, WarningConfig

    warnings = AutoHealthWarnings(config=WarningConfig())
    # Feed metrics each batch / epoch …
    alerts = warnings.check_batch(batch_metrics)
    alerts = warnings.check_epoch(epoch_metrics)
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

import numpy as np


@dataclass
class HealthWarning:
    """A single automated health warning."""
    rule_name: str
    severity: str              # "info" | "warning" | "danger" | "critical"
    message: str
    diagnosis: str
    recommendations: List[str] = field(default_factory=list)
    layer_name: Optional[str] = None
    epoch: Optional[int] = None
    batch: Optional[int] = None
    timestamp: float = field(default_factory=time.time)
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WarningConfig:
    """Configurable thresholds for the automated health warning system."""

    # Dead ReLU / zero activation
    dead_relu_zeros_pct: float = 50.0

    # Gradient norms
    gradient_vanish_thresh: float = 1e-7
    gradient_explode_thresh: float = 100.0
    gradient_spike_factor: float = 5.0
    gradient_rolling_window: int = 20

    # Overfitting
    overfit_patience: int = 3
    overfit_ratio: float = 1.3          # val_loss / train_loss

    # Stagnation
    weight_stagnation_eps: float = 1e-6
    weight_stagnation_window: int = 5
    loss_stagnation_eps: float = 1e-4
    loss_stagnation_window: int = 5

    # Divergence
    loss_divergence_window: int = 3

    # Activation saturation
    saturation_pct_thresh: float = 40.0

    # Learning rate heuristic
    lr_too_high_loss_factor: float = 3.0


class AutoHealthWarnings:
    """
    Stateful warning engine that tracks training metrics over time
    and fires threshold-based rules automatically.

    Attach to a training loop and call :meth:`check_batch` /
    :meth:`check_epoch` each iteration.  Accumulated warnings are
    accessible via :attr:`warnings`.
    """

    def __init__(self, config: Optional[WarningConfig] = None):
        self.config = config or WarningConfig()

        # Rolling state
        self._gradient_norms: Dict[str, Deque[float]] = {}
        self._weight_deltas: Dict[str, Deque[float]] = {}
        self._train_losses: List[float] = []
        self._val_losses: List[float] = []
        self._batch_losses: Deque[float] = deque(maxlen=100)

        # Collected warnings
        self._warnings: List[HealthWarning] = []

        # Dedup: avoid spamming the same warning every batch
        self._fired_rules: Dict[str, float] = {}  # rule_key → last-fired timestamp
        self._dedup_interval = 30.0  # seconds

    @property
    def warnings(self) -> List[HealthWarning]:
        return list(self._warnings)

    # ------------------------------------------------------------------
    # Per-batch check
    # ------------------------------------------------------------------

    def check_batch(
        self,
        *,
        epoch: Optional[int] = None,
        batch: Optional[int] = None,
        loss: Optional[float] = None,
        gradient_norms: Optional[Dict[str, float]] = None,
        weight_stats: Optional[Dict[str, Dict[str, Any]]] = None,
        activation_stats: Optional[Dict[str, Dict[str, Any]]] = None,
        weight_deltas: Optional[Dict[str, float]] = None,
    ) -> List[HealthWarning]:
        """Run all batch-level rules and return new warnings."""
        new: List[HealthWarning] = []

        if loss is not None:
            self._batch_losses.append(loss)
            new.extend(self._check_nan_inf_loss(loss, epoch, batch))
            new.extend(self._check_loss_divergence(epoch, batch))

        if gradient_norms:
            new.extend(self._check_gradients(gradient_norms, epoch, batch))

        if activation_stats:
            new.extend(self._check_activations(activation_stats, epoch, batch))

        if weight_stats:
            new.extend(self._check_dead_weights(weight_stats, epoch, batch))

        if weight_deltas:
            new.extend(self._check_weight_stagnation(weight_deltas, epoch, batch))

        self._warnings.extend(new)
        return new

    # ------------------------------------------------------------------
    # Per-epoch check
    # ------------------------------------------------------------------

    def check_epoch(
        self,
        *,
        epoch: int,
        train_loss: Optional[float] = None,
        val_loss: Optional[float] = None,
        train_acc: Optional[float] = None,
        val_acc: Optional[float] = None,
        gradient_norms: Optional[Dict[str, float]] = None,
        weight_stats: Optional[Dict[str, Dict[str, Any]]] = None,
        activation_stats: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> List[HealthWarning]:
        """Run all epoch-level rules and return new warnings."""
        new: List[HealthWarning] = []

        if train_loss is not None:
            self._train_losses.append(train_loss)
        if val_loss is not None:
            self._val_losses.append(val_loss)

        # Overfitting check
        new.extend(self._check_overfitting(epoch))

        # Loss stagnation
        new.extend(self._check_loss_stagnation(epoch))

        # Gradient checks (epoch-level too)
        if gradient_norms:
            new.extend(self._check_gradients(gradient_norms, epoch, None))

        # Activation / weight checks
        if activation_stats:
            new.extend(self._check_activations(activation_stats, epoch, None))
        if weight_stats:
            new.extend(self._check_dead_weights(weight_stats, epoch, None))

        self._warnings.extend(new)
        return new

    # ------------------------------------------------------------------
    # Rule implementations
    # ------------------------------------------------------------------

    def _should_fire(self, rule_key: str) -> bool:
        """De-duplicate: don't fire the same rule twice within the interval."""
        now = time.time()
        last = self._fired_rules.get(rule_key)
        if last is not None and (now - last) < self._dedup_interval:
            return False
        self._fired_rules[rule_key] = now
        return True

    # -- NaN / Inf --------------------------------------------------------

    def _check_nan_inf_loss(self, loss: float, epoch, batch) -> List[HealthWarning]:
        if not (np.isnan(loss) or np.isinf(loss)):
            return []
        key = "nan_inf_loss"
        if not self._should_fire(key):
            return []
        return [HealthWarning(
            rule_name="nan_inf_loss",
            severity="critical",
            message="NaN/Inf detected in loss!",
            diagnosis=(
                "Numerical instability has corrupted the loss. "
                "Training should be stopped immediately."
            ),
            recommendations=[
                "Lower the learning rate (current may be too high)",
                "Add gradient clipping (max_norm=1.0)",
                "Check input data for NaN/Inf values",
                "Use a more numerically stable loss function",
            ],
            epoch=epoch, batch=batch,
            data={"loss": float(loss) if np.isfinite(loss) else str(loss)},
        )]

    # -- Loss divergence --------------------------------------------------

    def _check_loss_divergence(self, epoch, batch) -> List[HealthWarning]:
        w = self.config.loss_divergence_window
        if len(self._batch_losses) < w:
            return []
        recent = list(self._batch_losses)[-w:]
        if recent[-1] > recent[0] * self.config.lr_too_high_loss_factor:
            key = "loss_divergence"
            if not self._should_fire(key):
                return []
            return [HealthWarning(
                rule_name="loss_divergence",
                severity="danger",
                message=f"Loss diverging over last {w} batches",
                diagnosis=(
                    "The loss is increasing rapidly, indicating training instability."
                ),
                recommendations=[
                    "Immediately lower the learning rate",
                    "Add gradient clipping",
                    "Check data preprocessing (normalise inputs)",
                ],
                epoch=epoch, batch=batch,
                data={"recent_losses": recent},
            )]
        return []

    # -- Gradient checks --------------------------------------------------

    def _check_gradients(self, gradient_norms: Dict[str, float],
                         epoch, batch) -> List[HealthWarning]:
        alerts: List[HealthWarning] = []
        cfg = self.config

        for layer, norm in gradient_norms.items():
            # NaN/Inf
            if np.isnan(norm) or np.isinf(norm):
                key = f"gradient_nan_{layer}"
                if self._should_fire(key):
                    alerts.append(HealthWarning(
                        rule_name="gradient_nan_inf",
                        severity="critical",
                        message=f"NaN/Inf gradient in '{layer}'",
                        diagnosis="Gradient corruption prevents learning.",
                        recommendations=[
                            "Lower the learning rate",
                            "Add gradient clipping (max_norm=1.0)",
                            "Use batch normalisation before this layer",
                        ],
                        layer_name=layer, epoch=epoch, batch=batch,
                    ))
                continue

            # Vanishing
            if norm < cfg.gradient_vanish_thresh:
                key = f"gradient_vanish_{layer}"
                if self._should_fire(key):
                    alerts.append(HealthWarning(
                        rule_name="vanishing_gradient",
                        severity="danger",
                        message=f"Vanishing gradient in '{layer}' (norm={norm:.2e})",
                        diagnosis="Gradients too small — this layer is effectively frozen.",
                        recommendations=[
                            "Switch to ReLU or LeakyReLU activation",
                            "Use batch normalisation",
                            "Try skip connections (ResNet-style)",
                        ],
                        layer_name=layer, epoch=epoch, batch=batch,
                        data={"norm": norm},
                    ))

            # Exploding
            if norm > cfg.gradient_explode_thresh:
                key = f"gradient_explode_{layer}"
                if self._should_fire(key):
                    alerts.append(HealthWarning(
                        rule_name="exploding_gradient",
                        severity="danger",
                        message=f"Exploding gradient in '{layer}' (norm={norm:.2e})",
                        diagnosis="Excessively large gradients cause unstable weight updates.",
                        recommendations=[
                            "Add gradient clipping (max_norm=1.0)",
                            "Lower the learning rate",
                            "Use batch normalisation",
                        ],
                        layer_name=layer, epoch=epoch, batch=batch,
                        data={"norm": norm},
                    ))

            # Spike
            buf = self._gradient_norms.setdefault(
                layer, deque(maxlen=cfg.gradient_rolling_window))
            if len(buf) >= 3:
                rolling_mean = float(np.mean(buf))
                if rolling_mean > 0 and norm > rolling_mean * cfg.gradient_spike_factor:
                    key = f"gradient_spike_{layer}"
                    if self._should_fire(key):
                        alerts.append(HealthWarning(
                            rule_name="gradient_spike",
                            severity="warning",
                            message=(f"Possible exploding gradient in '{layer}': "
                                     f"norm {norm:.2e} vs rolling avg {rolling_mean:.2e}"),
                            diagnosis="A sudden gradient spike may indicate instability.",
                            recommendations=[
                                "Add gradient clipping",
                                "Reduce learning rate temporarily",
                                "Check for outlier data in the current batch",
                            ],
                            layer_name=layer, epoch=epoch, batch=batch,
                            data={"norm": norm, "rolling_mean": rolling_mean},
                        ))
            buf.append(norm)

        return alerts

    # -- Activation checks ------------------------------------------------

    def _check_activations(self, activation_stats: Dict[str, Dict],
                           epoch, batch) -> List[HealthWarning]:
        alerts: List[HealthWarning] = []
        for layer, stats in activation_stats.items():
            zeros_pct = stats.get("zeros_pct", 0)
            act_type = stats.get("activation_type", "")

            # Dead ReLU
            if act_type in ("relu", "leaky_relu") and zeros_pct > self.config.dead_relu_zeros_pct:
                key = f"dead_relu_{layer}"
                if self._should_fire(key):
                    alerts.append(HealthWarning(
                        rule_name="dead_relu",
                        severity="warning",
                        message=f"Possible dying ReLU in '{layer}' ({zeros_pct:.1f}% zeros)",
                        diagnosis=(
                            "Neurons producing zero outputs will receive zero gradients "
                            "and never recover."
                        ),
                        recommendations=[
                            "Use LeakyReLU(negative_slope=0.01) instead of ReLU",
                            "Lower the learning rate",
                            "Use He initialisation",
                        ],
                        layer_name=layer, epoch=epoch, batch=batch,
                        data={"zeros_pct": zeros_pct},
                    ))

            # Saturation
            sat_pct = stats.get("saturation_pct", 0)
            if sat_pct > self.config.saturation_pct_thresh:
                key = f"saturation_{layer}"
                if self._should_fire(key):
                    alerts.append(HealthWarning(
                        rule_name="activation_saturation",
                        severity="warning",
                        message=f"{sat_pct:.1f}% activations saturated in '{layer}'",
                        diagnosis="Saturated activations produce near-zero gradients.",
                        recommendations=[
                            "Switch to ReLU or GELU activation",
                            "Normalise inputs to the layer",
                            "Use batch normalisation",
                        ],
                        layer_name=layer, epoch=epoch, batch=batch,
                        data={"saturation_pct": sat_pct},
                    ))
        return alerts

    # -- Weight checks ----------------------------------------------------

    def _check_dead_weights(self, weight_stats: Dict[str, Dict],
                            epoch, batch) -> List[HealthWarning]:
        alerts: List[HealthWarning] = []
        for layer, stats in weight_stats.items():
            zeros_pct = stats.get("zeros_pct", 0)
            if zeros_pct > self.config.dead_relu_zeros_pct:
                key = f"dead_weights_{layer}"
                if self._should_fire(key):
                    alerts.append(HealthWarning(
                        rule_name="dead_weights",
                        severity="warning",
                        message=f"{zeros_pct:.1f}% dead neurons in '{layer}'",
                        diagnosis="Most weights near zero — layer contributes nothing.",
                        recommendations=[
                            "Switch to LeakyReLU or ELU",
                            "Use a different weight initialisation",
                            "Lower the learning rate",
                        ],
                        layer_name=layer, epoch=epoch, batch=batch,
                        data={"zeros_pct": zeros_pct},
                    ))
        return alerts

    # -- Weight stagnation ------------------------------------------------

    def _check_weight_stagnation(self, weight_deltas: Dict[str, float],
                                 epoch, batch) -> List[HealthWarning]:
        alerts: List[HealthWarning] = []
        cfg = self.config
        for param, delta in weight_deltas.items():
            buf = self._weight_deltas.setdefault(
                param, deque(maxlen=cfg.weight_stagnation_window))
            buf.append(delta)
            if len(buf) >= cfg.weight_stagnation_window:
                if all(d < cfg.weight_stagnation_eps for d in buf):
                    key = f"weight_stagnation_{param}"
                    if self._should_fire(key):
                        alerts.append(HealthWarning(
                            rule_name="weight_stagnation",
                            severity="warning",
                            message=(f"Optimizer may have stagnated for '{param}' "
                                     f"({cfg.weight_stagnation_window} batches, "
                                     f"max Δ={max(buf):.2e})"),
                            diagnosis=(
                                "Weight updates are near-zero for several consecutive "
                                "batches, suggesting the optimizer has plateaued."
                            ),
                            recommendations=[
                                "Reduce learning rate and use a scheduler",
                                "Try a different optimizer (switch SGD↔Adam)",
                                "Check that gradients are flowing to this parameter",
                            ],
                            layer_name=param, epoch=epoch, batch=batch,
                            data={"max_delta": float(max(buf))},
                        ))
        return alerts

    # -- Overfitting ------------------------------------------------------

    def _check_overfitting(self, epoch: int) -> List[HealthWarning]:
        p = self.config.overfit_patience
        if len(self._train_losses) < p or len(self._val_losses) < p:
            return []
        recent_train = float(np.mean(self._train_losses[-p:]))
        recent_val = float(np.mean(self._val_losses[-p:]))
        if recent_train < 1e-12:
            return []
        ratio = recent_val / max(recent_train, 1e-12)
        if ratio > self.config.overfit_ratio:
            key = "overfitting"
            if not self._should_fire(key):
                return []
            return [HealthWarning(
                rule_name="overfitting",
                severity="warning",
                message=f"Possible overfitting (val/train loss ratio = {ratio:.2f})",
                diagnosis=(
                    "Validation loss is diverging from training loss, "
                    "indicating the model is memorising rather than learning."
                ),
                recommendations=[
                    "Add Dropout layers (rate=0.2-0.5)",
                    "Use L2 regularization (weight_decay=1e-4)",
                    "Get more training data or use data augmentation",
                    "Reduce model complexity (fewer layers / neurons)",
                ],
                epoch=epoch,
                data={"ratio": ratio, "train": recent_train, "val": recent_val},
            )]
        return []

    # -- Loss stagnation --------------------------------------------------

    def _check_loss_stagnation(self, epoch: int) -> List[HealthWarning]:
        w = self.config.loss_stagnation_window
        if len(self._train_losses) < w:
            return []
        recent = self._train_losses[-w:]
        delta = abs(recent[-1] - recent[0])
        if delta < self.config.loss_stagnation_eps:
            key = "loss_stagnation"
            if not self._should_fire(key):
                return []
            return [HealthWarning(
                rule_name="loss_stagnation",
                severity="warning",
                message=f"Loss stagnant for {w} epochs (Δ={delta:.2e})",
                diagnosis="Training progress has plateaued.",
                recommendations=[
                    "Reduce learning rate (try lr × 0.1)",
                    "Use learning rate scheduling (e.g., cosine annealing)",
                    "Try a different optimizer (switch SGD↔Adam)",
                ],
                epoch=epoch,
                data={"delta": delta, "window": w},
            )]
        return []

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_summary(self) -> Dict[str, Any]:
        """Return a structured summary of all warnings fired."""
        by_rule: Dict[str, int] = {}
        by_severity: Dict[str, int] = {}
        for w in self._warnings:
            by_rule[w.rule_name] = by_rule.get(w.rule_name, 0) + 1
            by_severity[w.severity] = by_severity.get(w.severity, 0) + 1
        return {
            "total_warnings": len(self._warnings),
            "by_rule": by_rule,
            "by_severity": by_severity,
            "warnings": [
                {
                    "rule": w.rule_name,
                    "severity": w.severity,
                    "message": w.message,
                    "layer": w.layer_name,
                    "epoch": w.epoch,
                    "batch": w.batch,
                }
                for w in self._warnings
            ],
        }

    def reset(self) -> None:
        """Clear all state."""
        self._warnings.clear()
        self._gradient_norms.clear()
        self._weight_deltas.clear()
        self._train_losses.clear()
        self._val_losses.clear()
        self._batch_losses.clear()
        self._fired_rules.clear()
