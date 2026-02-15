"""
Core Training Logger — Event-driven logging for Neurogebra Training Observatory.

Provides a structured, multi-level logger with event dispatching, colour-coded
terminal output, and pluggable backends (console, file, TensorBoard, …).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Log levels
# ---------------------------------------------------------------------------

class LogLevel(IntEnum):
    """Verbosity levels for the Training Observatory."""
    SILENT = 0
    BASIC = 1
    DETAILED = 2
    EXPERT = 3
    DEBUG = 4


# ---------------------------------------------------------------------------
# Log event
# ---------------------------------------------------------------------------

@dataclass
class LogEvent:
    """A single structured log event emitted during training."""
    event_type: str          # e.g. "epoch_start", "layer_forward", …
    level: LogLevel          # minimum level required to emit this event
    timestamp: float = field(default_factory=time.time)
    epoch: Optional[int] = None
    batch: Optional[int] = None
    layer_name: Optional[str] = None
    layer_index: Optional[int] = None
    data: Dict[str, Any] = field(default_factory=dict)
    severity: str = "info"   # "info", "warning", "danger", "success"
    message: str = ""


# ---------------------------------------------------------------------------
# Training Logger
# ---------------------------------------------------------------------------

class TrainingLogger:
    """
    Central Training Observatory logger.

    Receives events from trainers / models / monitors, filters by level,
    dispatches to registered backends (terminal, exporters, etc.).

    Usage::

        logger = TrainingLogger(level=LogLevel.EXPERT)
        logger.on_train_start(model_info={...})
        logger.on_epoch_start(epoch=0)
        ...
    """

    # Event type constants
    TRAIN_START = "train_start"
    TRAIN_END = "train_end"
    EPOCH_START = "epoch_start"
    EPOCH_END = "epoch_end"
    BATCH_START = "batch_start"
    BATCH_END = "batch_end"
    LAYER_FORWARD = "layer_forward"
    LAYER_BACKWARD = "layer_backward"
    GRADIENT_COMPUTED = "gradient_computed"
    WEIGHT_UPDATED = "weight_updated"
    HEALTH_CHECK = "health_check"
    PREDICTION = "prediction"

    def __init__(
        self,
        level: LogLevel = LogLevel.BASIC,
        backends: Optional[Sequence[Any]] = None,
    ):
        self.level = level
        self._backends: List[Any] = list(backends) if backends else []
        self._event_log: List[LogEvent] = []
        self._callbacks: Dict[str, List[Callable]] = {}
        self._start_time: Optional[float] = None
        self._epoch_start_time: Optional[float] = None
        self._batch_start_time: Optional[float] = None

        # Aggregate stats
        self.total_epochs: int = 0
        self.total_batches: int = 0
        self.current_epoch: int = 0
        self.current_batch: int = 0

    # ------------------------------------------------------------------
    # Backend management
    # ------------------------------------------------------------------

    def add_backend(self, backend: Any) -> "TrainingLogger":
        """Add a display/export backend (TerminalDisplay, JSONExporter …)."""
        self._backends.append(backend)
        return self

    def register_callback(self, event_type: str, fn: Callable) -> None:
        """Register a custom callback for a specific event type."""
        self._callbacks.setdefault(event_type, []).append(fn)

    # ------------------------------------------------------------------
    # Internal dispatch
    # ------------------------------------------------------------------

    def _emit(self, event: LogEvent) -> None:
        """Emit an event to all backends and callbacks."""
        if event.level > self.level:
            return
        self._event_log.append(event)

        for backend in self._backends:
            handler = getattr(backend, f"handle_{event.event_type}", None)
            if handler:
                handler(event)
            elif hasattr(backend, "handle_event"):
                backend.handle_event(event)

        for fn in self._callbacks.get(event.event_type, []):
            fn(event)

    # ------------------------------------------------------------------
    # High-level event helpers
    # ------------------------------------------------------------------

    def on_train_start(self, *, model_info: Optional[Dict] = None,
                       total_epochs: int = 0, total_samples: int = 0,
                       batch_size: int = 0, **extra) -> None:
        self._start_time = time.time()
        self.total_epochs = total_epochs
        self._emit(LogEvent(
            event_type=self.TRAIN_START,
            level=LogLevel.BASIC,
            data={
                "model_info": model_info or {},
                "total_epochs": total_epochs,
                "total_samples": total_samples,
                "batch_size": batch_size,
                **extra,
            },
            severity="info",
            message=f"Training started — {total_epochs} epochs, {total_samples} samples",
        ))

    def on_train_end(self, *, final_metrics: Optional[Dict] = None, **extra) -> None:
        elapsed = time.time() - self._start_time if self._start_time else 0
        self._emit(LogEvent(
            event_type=self.TRAIN_END,
            level=LogLevel.BASIC,
            data={
                "final_metrics": final_metrics or {},
                "total_time": elapsed,
                **extra,
            },
            severity="success",
            message=f"Training complete in {elapsed:.1f}s",
        ))

    def on_epoch_start(self, epoch: int, **extra) -> None:
        self.current_epoch = epoch
        self._epoch_start_time = time.time()
        self._emit(LogEvent(
            event_type=self.EPOCH_START,
            level=LogLevel.BASIC,
            epoch=epoch,
            data=extra,
            message=f"Epoch {epoch + 1}/{self.total_epochs}",
        ))

    def on_epoch_end(self, epoch: int, *, metrics: Optional[Dict] = None,
                     **extra) -> None:
        elapsed = time.time() - self._epoch_start_time if self._epoch_start_time else 0
        self._emit(LogEvent(
            event_type=self.EPOCH_END,
            level=LogLevel.BASIC,
            epoch=epoch,
            data={"metrics": metrics or {}, "epoch_time": elapsed, **extra},
            message=f"Epoch {epoch + 1} done — {elapsed:.2f}s",
        ))

    def on_batch_start(self, batch: int, *, epoch: Optional[int] = None,
                       X_batch: Optional[np.ndarray] = None,
                       y_batch: Optional[np.ndarray] = None, **extra) -> None:
        self.current_batch = batch
        self._batch_start_time = time.time()
        self._emit(LogEvent(
            event_type=self.BATCH_START,
            level=LogLevel.DETAILED,
            epoch=epoch,
            batch=batch,
            data={
                "batch_shape": X_batch.shape if X_batch is not None else None,
                **extra,
            },
        ))

    def on_batch_end(self, batch: int, *, epoch: Optional[int] = None,
                     loss: Optional[float] = None,
                     metrics: Optional[Dict] = None, **extra) -> None:
        elapsed = time.time() - self._batch_start_time if self._batch_start_time else 0
        self._emit(LogEvent(
            event_type=self.BATCH_END,
            level=LogLevel.DETAILED,
            epoch=epoch,
            batch=batch,
            data={"loss": loss, "metrics": metrics or {}, "batch_time": elapsed, **extra},
        ))

    def on_layer_forward(self, layer_index: int, layer_name: str, *,
                         input_data: Optional[np.ndarray] = None,
                         output_data: Optional[np.ndarray] = None,
                         weights: Optional[np.ndarray] = None,
                         bias: Optional[np.ndarray] = None,
                         formula: str = "", **extra) -> None:
        self._emit(LogEvent(
            event_type=self.LAYER_FORWARD,
            level=LogLevel.EXPERT,
            layer_name=layer_name,
            layer_index=layer_index,
            data={
                "input_shape": input_data.shape if input_data is not None else None,
                "output_shape": output_data.shape if output_data is not None else None,
                "input_stats": _array_stats(input_data),
                "output_stats": _array_stats(output_data),
                "weight_stats": _array_stats(weights),
                "bias_stats": _array_stats(bias),
                "formula": formula,
                **extra,
            },
        ))

    def on_layer_backward(self, layer_index: int, layer_name: str, *,
                          grad_output: Optional[np.ndarray] = None,
                          grad_weights: Optional[np.ndarray] = None,
                          grad_bias: Optional[np.ndarray] = None,
                          formula: str = "", **extra) -> None:
        self._emit(LogEvent(
            event_type=self.LAYER_BACKWARD,
            level=LogLevel.EXPERT,
            layer_name=layer_name,
            layer_index=layer_index,
            data={
                "grad_output_stats": _array_stats(grad_output),
                "grad_weights_stats": _array_stats(grad_weights),
                "grad_bias_stats": _array_stats(grad_bias),
                "formula": formula,
                **extra,
            },
        ))

    def on_gradient_computed(self, param_name: str, gradient: float, **extra) -> None:
        sev = "info"
        if abs(gradient) < 1e-7:
            sev = "danger"
        elif abs(gradient) > 100:
            sev = "danger"
        self._emit(LogEvent(
            event_type=self.GRADIENT_COMPUTED,
            level=LogLevel.EXPERT,
            data={"param": param_name, "gradient": gradient, **extra},
            severity=sev,
        ))

    def on_weight_updated(self, param_name: str, old_value: float,
                          new_value: float, **extra) -> None:
        self._emit(LogEvent(
            event_type=self.WEIGHT_UPDATED,
            level=LogLevel.EXPERT,
            data={"param": param_name, "old": old_value, "new": new_value, **extra},
        ))

    def on_health_check(self, check_name: str, severity: str,
                        message: str, recommendations: Optional[List[str]] = None,
                        **extra) -> None:
        self._emit(LogEvent(
            event_type=self.HEALTH_CHECK,
            level=LogLevel.BASIC,
            data={"check": check_name, "recommendations": recommendations or [], **extra},
            severity=severity,
            message=message,
        ))

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_event_log(self) -> List[LogEvent]:
        """Return all recorded events."""
        return list(self._event_log)

    def get_events_by_type(self, event_type: str) -> List[LogEvent]:
        return [e for e in self._event_log if e.event_type == event_type]

    def summary(self) -> Dict[str, Any]:
        """Return aggregate statistics for the training run."""
        events = self._event_log
        return {
            "total_events": len(events),
            "event_types": list({e.event_type for e in events}),
            "danger_events": sum(1 for e in events if e.severity == "danger"),
            "warning_events": sum(1 for e in events if e.severity == "warning"),
            "total_epochs": self.total_epochs,
        }

    def clear(self) -> None:
        self._event_log.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _array_stats(arr: Optional[np.ndarray]) -> Optional[Dict[str, float]]:
    """Compute summary statistics of a numpy array."""
    if arr is None:
        return None
    flat = np.asarray(arr, dtype=np.float64).ravel()
    if flat.size == 0:
        return None
    return {
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "norm_l2": float(np.linalg.norm(flat)),
        "zeros_pct": float(np.sum(flat == 0) / flat.size * 100),
        "nan_count": int(np.sum(np.isnan(flat))),
        "inf_count": int(np.sum(np.isinf(flat))),
        "size": int(flat.size),
    }
