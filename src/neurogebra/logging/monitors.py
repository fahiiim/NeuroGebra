"""
Monitors — Real-time gradient, weight, activation and performance tracking.

Each monitor analyses arrays produced during forward/backward passes and
emits structured diagnostics through the TrainingLogger.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


# ======================================================================
# Gradient Monitor
# ======================================================================

class GradientMonitor:
    """
    Track gradient norms, detect vanishing / exploding gradients,
    and compute gradient-to-weight ratios per layer per epoch.
    """

    VANISHING_THRESHOLD = 1e-7
    EXPLODING_THRESHOLD = 100.0

    def __init__(self):
        self.history: Dict[str, List[Dict[str, float]]] = {}  # layer → list of stats

    def record(self, layer_name: str, gradient: np.ndarray,
               weights: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Record gradient statistics for a layer and return a report dict."""
        g = np.asarray(gradient, dtype=np.float64).ravel()
        stats: Dict[str, Any] = {
            "norm_l2": float(np.linalg.norm(g)),
            "norm_l1": float(np.sum(np.abs(g))),
            "mean": float(np.mean(g)),
            "std": float(np.std(g)),
            "min": float(np.min(g)),
            "max": float(np.max(g)),
            "nan_count": int(np.sum(np.isnan(g))),
            "inf_count": int(np.sum(np.isinf(g))),
        }

        # Gradient-to-weight ratio
        if weights is not None:
            w = np.asarray(weights, dtype=np.float64).ravel()
            w_norm = float(np.linalg.norm(w))
            stats["grad_weight_ratio"] = stats["norm_l2"] / max(w_norm, 1e-12)
        else:
            stats["grad_weight_ratio"] = None

        # Diagnosis
        stats["status"] = "healthy"
        stats["alerts"] = []

        if stats["nan_count"] > 0 or stats["inf_count"] > 0:
            stats["status"] = "critical"
            stats["alerts"].append("NaN/Inf detected in gradients!")

        elif stats["norm_l2"] < self.VANISHING_THRESHOLD:
            stats["status"] = "danger"
            stats["alerts"].append(
                f"Vanishing gradient (L2 norm={stats['norm_l2']:.2e})"
            )
        elif stats["norm_l2"] > self.EXPLODING_THRESHOLD:
            stats["status"] = "danger"
            stats["alerts"].append(
                f"Exploding gradient (L2 norm={stats['norm_l2']:.2e})"
            )

        self.history.setdefault(layer_name, []).append(stats)
        return stats


# ======================================================================
# Weight Monitor
# ======================================================================

class WeightMonitor:
    """Track weight distributions, dead neurons, and symmetry."""

    DEAD_THRESHOLD = 1e-6
    DEAD_NEURON_PCT_WARN = 50.0

    def __init__(self):
        self.history: Dict[str, List[Dict[str, float]]] = {}

    def record(self, layer_name: str, weights: np.ndarray) -> Dict[str, Any]:
        w = np.asarray(weights, dtype=np.float64).ravel()
        stats: Dict[str, Any] = {
            "mean": float(np.mean(w)),
            "std": float(np.std(w)),
            "min": float(np.min(w)),
            "max": float(np.max(w)),
            "norm_l2": float(np.linalg.norm(w)),
            "zeros_pct": float(np.sum(np.abs(w) < self.DEAD_THRESHOLD) / w.size * 100),
            "size": int(w.size),
        }

        # Histogram (10 bins)
        counts, edges = np.histogram(w, bins=10)
        stats["histogram"] = {
            "counts": counts.tolist(),
            "edges": edges.tolist(),
        }

        # Alerts
        stats["status"] = "healthy"
        stats["alerts"] = []
        if stats["zeros_pct"] > self.DEAD_NEURON_PCT_WARN:
            stats["status"] = "warning"
            stats["alerts"].append(
                f"{stats['zeros_pct']:.1f}% of weights near zero – possible dead neurons"
            )

        # Weight change tracking
        prev = self.history.get(layer_name)
        if prev:
            prev_norm = prev[-1]["norm_l2"]
            delta = abs(stats["norm_l2"] - prev_norm) / max(prev_norm, 1e-12)
            stats["change_pct"] = delta * 100
        else:
            stats["change_pct"] = 0.0

        self.history.setdefault(layer_name, []).append(stats)
        return stats


# ======================================================================
# Activation Monitor
# ======================================================================

class ActivationMonitor:
    """Track activation statistics, saturation, and dead units."""

    SATURATION_LOW = 0.01
    SATURATION_HIGH = 0.99

    def __init__(self):
        self.history: Dict[str, List[Dict[str, float]]] = {}

    def record(self, layer_name: str, activations: np.ndarray,
               activation_type: str = "unknown") -> Dict[str, Any]:
        a = np.asarray(activations, dtype=np.float64).ravel()
        stats: Dict[str, Any] = {
            "mean": float(np.mean(a)),
            "std": float(np.std(a)),
            "min": float(np.min(a)),
            "max": float(np.max(a)),
            "zeros_pct": float(np.sum(a == 0) / a.size * 100),
            "size": int(a.size),
            "activation_type": activation_type,
        }

        stats["status"] = "healthy"
        stats["alerts"] = []

        # Dead ReLU check
        if activation_type in ("relu", "leaky_relu"):
            if stats["zeros_pct"] > 50:
                stats["status"] = "warning"
                stats["alerts"].append(
                    f"{stats['zeros_pct']:.1f}% zeros — potential dead ReLU neurons"
                )

        # Sigmoid/tanh saturation check
        if activation_type in ("sigmoid", "tanh"):
            if activation_type == "sigmoid":
                sat = float(np.sum((a < self.SATURATION_LOW) | (a > self.SATURATION_HIGH)) / a.size * 100)
            else:
                sat = float(np.sum((a < -0.99) | (a > 0.99)) / a.size * 100)
            stats["saturation_pct"] = sat
            if sat > 40:
                stats["status"] = "warning"
                stats["alerts"].append(
                    f"{sat:.1f}% activations are saturated"
                )

        self.history.setdefault(layer_name, []).append(stats)
        return stats


# ======================================================================
# Performance Monitor
# ======================================================================

class PerformanceMonitor:
    """Track per-layer computation time, memory, and bottlenecks."""

    def __init__(self):
        self.layer_times: Dict[str, List[float]] = {}   # forward
        self.backward_times: Dict[str, List[float]] = {}
        self.epoch_times: List[float] = []
        self.batch_times: List[float] = []
        self._memory_snapshots: List[Dict[str, int]] = []

    def record_layer_time(self, layer_name: str, elapsed: float,
                          phase: str = "forward") -> None:
        bucket = self.layer_times if phase == "forward" else self.backward_times
        bucket.setdefault(layer_name, []).append(elapsed)

    def record_epoch_time(self, elapsed: float) -> None:
        self.epoch_times.append(elapsed)

    def record_batch_time(self, elapsed: float) -> None:
        self.batch_times.append(elapsed)

    def bottleneck_report(self) -> Dict[str, Any]:
        """Identify the layer consuming the most time."""
        if not self.layer_times:
            return {"bottleneck": None}
        avg_times = {
            name: float(np.mean(times))
            for name, times in self.layer_times.items()
        }
        worst = max(avg_times, key=avg_times.get)  # type: ignore[arg-type]
        return {
            "bottleneck": worst,
            "avg_time_ms": avg_times[worst] * 1000,
            "all_layers": {k: v * 1000 for k, v in avg_times.items()},
        }

    def summary(self) -> Dict[str, Any]:
        return {
            "total_epoch_time": sum(self.epoch_times),
            "avg_epoch_time": float(np.mean(self.epoch_times)) if self.epoch_times else 0,
            "avg_batch_time": float(np.mean(self.batch_times)) if self.batch_times else 0,
            "bottleneck": self.bottleneck_report(),
        }
