"""
AdaptiveLogger — Smart anomaly-triggered logging for the Training Observatory.

Instead of unconditionally logging everything at EXPERT level, the adaptive
logger stays at BASIC level and **only** escalates to EXPERT detail when
something actually looks suspicious.  This can reduce log size by 80-90 %
while keeping all diagnostic value.

Anomaly triggers (configurable):
    - ``zeros_pct`` crosses a threshold (default 50 %)
    - Gradient L2 norm spikes by more than ``spike_factor`` × rolling average
    - Loss increases between consecutive batches by > ``loss_spike_pct`` %
    - NaN or Inf appears anywhere
    - Activation saturation exceeds threshold

Usage::

    from neurogebra.logging.adaptive import AdaptiveLogger, AnomalyConfig
    from neurogebra.logging.logger import TrainingLogger, LogLevel

    base_logger = TrainingLogger(level=LogLevel.EXPERT)
    adaptive = AdaptiveLogger(base_logger, config=AnomalyConfig())

    # Use *adaptive* as a drop-in replacement where you'd call base_logger.
    adaptive.on_train_start(model_info={...})
    adaptive.on_layer_forward(...)   # silenced unless anomaly detected
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence

import numpy as np

from neurogebra.logging.logger import LogEvent, LogLevel, TrainingLogger


# ---------------------------------------------------------------------------
# Anomaly configuration
# ---------------------------------------------------------------------------

@dataclass
class AnomalyConfig:
    """Thresholds that trigger escalation from BASIC → EXPERT logging."""

    # Dead neuron / zero activation threshold (percent)
    zeros_pct_threshold: float = 50.0

    # Gradient spike: current norm > rolling_mean × spike_factor
    gradient_spike_factor: float = 5.0
    gradient_rolling_window: int = 20

    # Gradient absolute thresholds
    gradient_vanish_threshold: float = 1e-7
    gradient_explode_threshold: float = 100.0

    # Loss spike between consecutive batches (percent increase)
    loss_spike_pct: float = 50.0

    # Activation saturation threshold (percent)
    saturation_threshold: float = 40.0

    # Weight delta near-zero (consecutive batches)
    weight_stagnation_threshold: float = 1e-6
    weight_stagnation_window: int = 5

    # How many events to keep in "escalated" mode after an anomaly
    escalation_cooldown: int = 10


# ---------------------------------------------------------------------------
# Anomaly record
# ---------------------------------------------------------------------------

@dataclass
class AnomalyRecord:
    """Structured record of a detected anomaly."""
    anomaly_type: str          # e.g. "dead_relu", "gradient_spike", ...
    timestamp: float = field(default_factory=time.time)
    epoch: Optional[int] = None
    batch: Optional[int] = None
    layer_name: Optional[str] = None
    severity: str = "warning"  # "warning" | "danger" | "critical"
    message: str = ""
    data: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Adaptive Logger
# ---------------------------------------------------------------------------

class AdaptiveLogger:
    """
    Wraps a :class:`TrainingLogger` and filters events adaptively.

    In **normal** mode only BASIC-level events are emitted.
    When an anomaly is detected the logger temporarily escalates to EXPERT
    for ``escalation_cooldown`` events, so the user gets the full picture
    around the anomaly without drowning in noise the rest of the time.

    The underlying ``TrainingLogger`` must be created with
    ``level=LogLevel.EXPERT`` (or higher) so it *can* emit the detailed
    events when the adaptive logger un-mutes them.
    """

    def __init__(
        self,
        base_logger: TrainingLogger,
        config: Optional[AnomalyConfig] = None,
    ):
        self._base = base_logger
        self.config = config or AnomalyConfig()

        # Ensure the base logger will accept EXPERT events
        if self._base.level < LogLevel.EXPERT:
            self._base.level = LogLevel.EXPERT

        # Rolling state
        self._gradient_norms: Dict[str, Deque[float]] = {}
        self._last_batch_loss: Optional[float] = None
        self._weight_deltas: Dict[str, Deque[float]] = {}
        self._anomalies: List[AnomalyRecord] = []

        # Escalation bookkeeping
        self._escalated = False
        self._escalation_counter = 0

        # Shadow level: the level we *pretend* the logger is at
        self._effective_level = LogLevel.BASIC

    # ------------------------------------------------------------------
    # Public API — mirrors TrainingLogger
    # ------------------------------------------------------------------

    @property
    def anomalies(self) -> List[AnomalyRecord]:
        """Return all detected anomalies so far."""
        return list(self._anomalies)

    @property
    def is_escalated(self) -> bool:
        return self._escalated

    # Delegate attribute access to the base logger for anything not overridden
    def __getattr__(self, name: str):
        return getattr(self._base, name)

    # -- train lifecycle --------------------------------------------------

    def on_train_start(self, **kwargs) -> None:
        self._base.on_train_start(**kwargs)

    def on_train_end(self, **kwargs) -> None:
        self._base.on_train_end(**kwargs)

    def on_epoch_start(self, epoch: int, **kwargs) -> None:
        self._base.on_epoch_start(epoch, **kwargs)

    def on_epoch_end(self, epoch: int, **kwargs) -> None:
        self._base.on_epoch_end(epoch, **kwargs)

    def on_batch_start(self, batch: int, **kwargs) -> None:
        self._base.on_batch_start(batch, **kwargs)

    def on_batch_end(self, batch: int, **kwargs) -> None:
        loss = kwargs.get("loss")
        if loss is not None:
            self._check_loss_spike(loss, kwargs.get("epoch"), batch)
            self._last_batch_loss = loss
        self._base.on_batch_end(batch, **kwargs)

    # -- layer-level (gated) -----------------------------------------------

    def on_layer_forward(self, layer_index: int, layer_name: str, **kwargs) -> None:
        """Only emit EXPERT-level layer_forward when escalated or anomalous."""
        anomaly = self._check_forward_anomaly(layer_name, kwargs)
        if anomaly or self._escalated:
            self._base.on_layer_forward(layer_index, layer_name, **kwargs)
        # else: silently skip

    def on_layer_backward(self, layer_index: int, layer_name: str, **kwargs) -> None:
        anomaly = self._check_backward_anomaly(layer_name, kwargs)
        if anomaly or self._escalated:
            self._base.on_layer_backward(layer_index, layer_name, **kwargs)

    def on_gradient_computed(self, param_name: str, gradient: float, **kwargs) -> None:
        anomaly = self._check_gradient_anomaly(param_name, gradient)
        if anomaly or self._escalated:
            self._base.on_gradient_computed(param_name, gradient, **kwargs)

    def on_weight_updated(self, param_name: str, old_value: float,
                          new_value: float, **kwargs) -> None:
        anomaly = self._check_weight_stagnation(param_name, old_value, new_value)
        if anomaly or self._escalated:
            self._base.on_weight_updated(param_name, old_value, new_value, **kwargs)

    def on_health_check(self, *args, **kwargs) -> None:
        self._base.on_health_check(*args, **kwargs)

    # ------------------------------------------------------------------
    # Anomaly detection helpers
    # ------------------------------------------------------------------

    def _flag_anomaly(self, record: AnomalyRecord) -> None:
        """Register an anomaly and enter escalated mode."""
        self._anomalies.append(record)
        self._escalated = True
        self._escalation_counter = self.config.escalation_cooldown

        # Also emit a health-check event
        self._base.on_health_check(
            check_name=f"adaptive_{record.anomaly_type}",
            severity=record.severity,
            message=record.message,
            recommendations=[],
            anomaly_data=record.data,
        )

    def _tick_escalation(self) -> None:
        """Count down the escalation cooldown after each gated event."""
        if self._escalated:
            self._escalation_counter -= 1
            if self._escalation_counter <= 0:
                self._escalated = False

    # -- forward checks ---------------------------------------------------

    def _check_forward_anomaly(self, layer_name: str, kwargs: Dict) -> bool:
        self._tick_escalation()

        output_data = kwargs.get("output_data")
        if output_data is not None:
            arr = np.asarray(output_data, dtype=np.float64)

            # NaN / Inf
            if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
                self._flag_anomaly(AnomalyRecord(
                    anomaly_type="nan_inf_activation",
                    layer_name=layer_name,
                    severity="critical",
                    message=f"NaN/Inf detected in activations of '{layer_name}'",
                    data={"nan_count": int(np.sum(np.isnan(arr))),
                          "inf_count": int(np.sum(np.isinf(arr)))},
                ))
                return True

            # Dead neurons
            flat = arr.ravel()
            zeros_pct = float(np.sum(flat == 0) / max(flat.size, 1) * 100)
            if zeros_pct > self.config.zeros_pct_threshold:
                self._flag_anomaly(AnomalyRecord(
                    anomaly_type="dead_neurons",
                    layer_name=layer_name,
                    severity="warning",
                    message=(f"{zeros_pct:.1f}% zeros in '{layer_name}' "
                             f"— possible dying ReLU"),
                    data={"zeros_pct": zeros_pct},
                ))
                return True

            # Saturation (sigmoid/tanh activations mostly ∈ (0,1) or (-1,1))
            if flat.size > 0:
                sat_low = float(np.sum(np.abs(flat) < 0.01) / flat.size * 100)
                sat_high = float(np.sum(np.abs(flat) > 0.99) / flat.size * 100)
                sat_total = sat_low + sat_high
                if sat_total > self.config.saturation_threshold:
                    self._flag_anomaly(AnomalyRecord(
                        anomaly_type="activation_saturation",
                        layer_name=layer_name,
                        severity="warning",
                        message=(f"{sat_total:.1f}% activations saturated "
                                 f"in '{layer_name}'"),
                        data={"saturation_pct": sat_total},
                    ))
                    return True

        return False

    # -- backward checks --------------------------------------------------

    def _check_backward_anomaly(self, layer_name: str, kwargs: Dict) -> bool:
        self._tick_escalation()
        grad_output = kwargs.get("grad_output")
        if grad_output is not None:
            arr = np.asarray(grad_output, dtype=np.float64)
            if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
                self._flag_anomaly(AnomalyRecord(
                    anomaly_type="nan_inf_gradient",
                    layer_name=layer_name,
                    severity="critical",
                    message=f"NaN/Inf in gradient output of '{layer_name}'",
                ))
                return True
        return False

    # -- gradient norm checks ---------------------------------------------

    def _check_gradient_anomaly(self, param_name: str, gradient: float) -> bool:
        self._tick_escalation()
        g = abs(gradient)

        # Absolute thresholds
        if g < self.config.gradient_vanish_threshold:
            self._flag_anomaly(AnomalyRecord(
                anomaly_type="vanishing_gradient",
                layer_name=param_name,
                severity="danger",
                message=f"Vanishing gradient for '{param_name}' (|g|={g:.2e})",
                data={"gradient": gradient},
            ))
            return True

        if g > self.config.gradient_explode_threshold:
            self._flag_anomaly(AnomalyRecord(
                anomaly_type="exploding_gradient",
                layer_name=param_name,
                severity="danger",
                message=f"Exploding gradient for '{param_name}' (|g|={g:.2e})",
                data={"gradient": gradient},
            ))
            return True

        # Spike detection
        buf = self._gradient_norms.setdefault(
            param_name, deque(maxlen=self.config.gradient_rolling_window)
        )
        if len(buf) >= 3:
            rolling_mean = float(np.mean(buf))
            if rolling_mean > 0 and g > rolling_mean * self.config.gradient_spike_factor:
                self._flag_anomaly(AnomalyRecord(
                    anomaly_type="gradient_spike",
                    layer_name=param_name,
                    severity="warning",
                    message=(f"Gradient spike in '{param_name}': "
                             f"{g:.2e} vs rolling avg {rolling_mean:.2e}"),
                    data={"gradient": gradient, "rolling_mean": rolling_mean},
                ))
                buf.append(g)
                return True
        buf.append(g)
        return False

    # -- loss spike -------------------------------------------------------

    def _check_loss_spike(self, loss: float, epoch: Optional[int],
                          batch: int) -> bool:
        if self._last_batch_loss is not None and self._last_batch_loss > 0:
            pct_increase = (loss - self._last_batch_loss) / self._last_batch_loss * 100
            if pct_increase > self.config.loss_spike_pct:
                self._flag_anomaly(AnomalyRecord(
                    anomaly_type="loss_spike",
                    epoch=epoch,
                    batch=batch,
                    severity="warning",
                    message=(f"Loss spiked by {pct_increase:.1f}% "
                             f"({self._last_batch_loss:.4f} → {loss:.4f})"),
                    data={"prev_loss": self._last_batch_loss, "new_loss": loss,
                          "pct_increase": pct_increase},
                ))
                return True
        return False

    # -- weight stagnation ------------------------------------------------

    def _check_weight_stagnation(self, param_name: str, old: float,
                                 new: float) -> bool:
        self._tick_escalation()
        delta = abs(new - old)
        buf = self._weight_deltas.setdefault(
            param_name, deque(maxlen=self.config.weight_stagnation_window)
        )
        buf.append(delta)
        if len(buf) >= self.config.weight_stagnation_window:
            if all(d < self.config.weight_stagnation_threshold for d in buf):
                self._flag_anomaly(AnomalyRecord(
                    anomaly_type="weight_stagnation",
                    layer_name=param_name,
                    severity="warning",
                    message=(f"Weight '{param_name}' stagnant for "
                             f"{self.config.weight_stagnation_window} updates "
                             f"(max Δ={max(buf):.2e})"),
                    data={"max_delta": float(max(buf))},
                ))
                return True
        return False

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_anomaly_summary(self) -> Dict[str, Any]:
        """Return a structured summary of all detected anomalies."""
        by_type: Dict[str, int] = {}
        by_severity: Dict[str, int] = {}
        for a in self._anomalies:
            by_type[a.anomaly_type] = by_type.get(a.anomaly_type, 0) + 1
            by_severity[a.severity] = by_severity.get(a.severity, 0) + 1
        return {
            "total_anomalies": len(self._anomalies),
            "by_type": by_type,
            "by_severity": by_severity,
            "anomalies": [
                {
                    "type": a.anomaly_type,
                    "severity": a.severity,
                    "message": a.message,
                    "layer": a.layer_name,
                    "epoch": a.epoch,
                    "batch": a.batch,
                    "timestamp": a.timestamp,
                }
                for a in self._anomalies
            ],
        }

    def reset(self) -> None:
        """Clear all anomaly state and go back to BASIC mode."""
        self._anomalies.clear()
        self._gradient_norms.clear()
        self._weight_deltas.clear()
        self._last_batch_loss = None
        self._escalated = False
        self._escalation_counter = 0
