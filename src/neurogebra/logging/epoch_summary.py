"""
EpochSummarizer — Aggregate batch-level stats into per-epoch statistical summaries.

After every epoch the summarizer computes mean, std, min, max across all
batches in that epoch for every tracked metric, then emits a structured
summary event through the ``TrainingLogger``.

Usage::

    from neurogebra.logging.epoch_summary import EpochSummarizer

    summarizer = EpochSummarizer()

    for batch in batches:
        summarizer.record_batch(
            epoch=epoch,
            metrics={"loss": loss, "accuracy": acc},
            gradient_norms={"dense_0": 0.12, "dense_1": 0.03},
            weight_stats={"dense_0": {"mean": 0.01, "std": 0.1}},
            activation_stats={"dense_0": {"zeros_pct": 12.0}},
        )

    summary = summarizer.finalize_epoch(epoch)
    print(summary)
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class EpochStats:
    """Statistical summary of one metric across all batches in an epoch."""
    name: str
    count: int
    mean: float
    std: float
    min: float
    max: float
    first: float
    last: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "count": self.count,
            "mean": round(self.mean, 6),
            "std": round(self.std, 6),
            "min": round(self.min, 6),
            "max": round(self.max, 6),
            "first": round(self.first, 6),
            "last": round(self.last, 6),
        }


@dataclass
class EpochSummary:
    """Full per-epoch summary across all tracked dimensions."""
    epoch: int
    num_batches: int
    metrics: Dict[str, EpochStats] = field(default_factory=dict)
    gradient_norms: Dict[str, EpochStats] = field(default_factory=dict)
    weight_summaries: Dict[str, Dict[str, EpochStats]] = field(default_factory=dict)
    activation_summaries: Dict[str, Dict[str, EpochStats]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "epoch": self.epoch,
            "num_batches": self.num_batches,
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
            "gradient_norms": {k: v.to_dict() for k, v in self.gradient_norms.items()},
            "weight_summaries": {
                layer: {k: v.to_dict() for k, v in stats.items()}
                for layer, stats in self.weight_summaries.items()
            },
            "activation_summaries": {
                layer: {k: v.to_dict() for k, v in stats.items()}
                for layer, stats in self.activation_summaries.items()
            },
        }

    def format_text(self) -> str:
        """Return a human-readable text summary."""
        lines = [f"══ Epoch {self.epoch + 1} Summary ({self.num_batches} batches) ══"]

        if self.metrics:
            lines.append("  Metrics:")
            for name, s in self.metrics.items():
                lines.append(
                    f"    {name:20s}  mean={s.mean:.6f}  std={s.std:.6f}  "
                    f"min={s.min:.6f}  max={s.max:.6f}"
                )

        if self.gradient_norms:
            lines.append("  Gradient Norms:")
            for layer, s in self.gradient_norms.items():
                lines.append(
                    f"    {layer:20s}  mean={s.mean:.4e}  std={s.std:.4e}  "
                    f"min={s.min:.4e}  max={s.max:.4e}"
                )

        if self.weight_summaries:
            lines.append("  Weight Stats:")
            for layer, stats_map in self.weight_summaries.items():
                parts = []
                for key, s in stats_map.items():
                    parts.append(f"{key}: mean={s.mean:.4e}")
                lines.append(f"    {layer:20s}  " + "  ".join(parts))

        if self.activation_summaries:
            lines.append("  Activation Stats:")
            for layer, stats_map in self.activation_summaries.items():
                parts = []
                for key, s in stats_map.items():
                    parts.append(f"{key}: mean={s.mean:.4e}")
                lines.append(f"    {layer:20s}  " + "  ".join(parts))

        return "\n".join(lines)


class EpochSummarizer:
    """
    Accumulates batch-level data and produces per-epoch statistical summaries.

    Call :meth:`record_batch` for every batch, then :meth:`finalize_epoch`
    at the end of the epoch to get an :class:`EpochSummary`.
    """

    def __init__(self):
        # {epoch: {metric_name: [values]}}
        self._metric_buffers: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        # {epoch: {layer: [norm_values]}}
        self._gradient_buffers: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        # {epoch: {layer: {stat_name: [values]}}}
        self._weight_buffers: Dict[int, Dict[str, Dict[str, List[float]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )
        self._activation_buffers: Dict[int, Dict[str, Dict[str, List[float]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )
        self._batch_counts: Dict[int, int] = defaultdict(int)
        self._summaries: List[EpochSummary] = []

    @property
    def summaries(self) -> List[EpochSummary]:
        return list(self._summaries)

    def record_batch(
        self,
        epoch: int,
        *,
        metrics: Optional[Dict[str, float]] = None,
        gradient_norms: Optional[Dict[str, float]] = None,
        weight_stats: Optional[Dict[str, Dict[str, float]]] = None,
        activation_stats: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> None:
        """Buffer one batch of data for the given epoch."""
        self._batch_counts[epoch] += 1

        if metrics:
            buf = self._metric_buffers[epoch]
            for key, val in metrics.items():
                if isinstance(val, (int, float)) and np.isfinite(val):
                    buf[key].append(float(val))

        if gradient_norms:
            buf = self._gradient_buffers[epoch]
            for layer, norm in gradient_norms.items():
                if np.isfinite(norm):
                    buf[layer].append(float(norm))

        if weight_stats:
            buf = self._weight_buffers[epoch]
            for layer, stats in weight_stats.items():
                for key, val in stats.items():
                    if isinstance(val, (int, float)) and np.isfinite(val):
                        buf[layer][key].append(float(val))

        if activation_stats:
            buf = self._activation_buffers[epoch]
            for layer, stats in activation_stats.items():
                for key, val in stats.items():
                    if isinstance(val, (int, float)) and np.isfinite(val):
                        buf[layer][key].append(float(val))

    def finalize_epoch(self, epoch: int) -> EpochSummary:
        """
        Compute and return the statistical summary for *epoch*.

        Automatically clears batch buffers for that epoch.
        """
        n_batches = self._batch_counts.get(epoch, 0)

        # Metrics
        metric_stats: Dict[str, EpochStats] = {}
        for name, vals in self._metric_buffers.get(epoch, {}).items():
            if vals:
                metric_stats[name] = _compute_stats(name, vals)

        # Gradient norms
        grad_stats: Dict[str, EpochStats] = {}
        for layer, vals in self._gradient_buffers.get(epoch, {}).items():
            if vals:
                grad_stats[layer] = _compute_stats(layer, vals)

        # Weight summaries
        weight_sums: Dict[str, Dict[str, EpochStats]] = {}
        for layer, keys in self._weight_buffers.get(epoch, {}).items():
            weight_sums[layer] = {}
            for key, vals in keys.items():
                if vals:
                    weight_sums[layer][key] = _compute_stats(key, vals)

        # Activation summaries
        act_sums: Dict[str, Dict[str, EpochStats]] = {}
        for layer, keys in self._activation_buffers.get(epoch, {}).items():
            act_sums[layer] = {}
            for key, vals in keys.items():
                if vals:
                    act_sums[layer][key] = _compute_stats(key, vals)

        summary = EpochSummary(
            epoch=epoch,
            num_batches=n_batches,
            metrics=metric_stats,
            gradient_norms=grad_stats,
            weight_summaries=weight_sums,
            activation_summaries=act_sums,
        )
        self._summaries.append(summary)

        # Cleanup
        self._metric_buffers.pop(epoch, None)
        self._gradient_buffers.pop(epoch, None)
        self._weight_buffers.pop(epoch, None)
        self._activation_buffers.pop(epoch, None)
        self._batch_counts.pop(epoch, None)

        return summary

    def get_all_summaries(self) -> List[Dict[str, Any]]:
        """Return all epoch summaries as dicts."""
        return [s.to_dict() for s in self._summaries]

    def reset(self) -> None:
        """Clear all state."""
        self._metric_buffers.clear()
        self._gradient_buffers.clear()
        self._weight_buffers.clear()
        self._activation_buffers.clear()
        self._batch_counts.clear()
        self._summaries.clear()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _compute_stats(name: str, values: List[float]) -> EpochStats:
    arr = np.array(values, dtype=np.float64)
    return EpochStats(
        name=name,
        count=len(arr),
        mean=float(np.mean(arr)),
        std=float(np.std(arr)),
        min=float(np.min(arr)),
        max=float(np.max(arr)),
        first=float(arr[0]),
        last=float(arr[-1]),
    )
