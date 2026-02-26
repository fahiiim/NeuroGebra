"""
TieredStorage — Separate training logs into concern-specific streams/files.

Instead of one flat JSON array of 77k entries, the tiered storage system
writes three separate log files:

- **basic.log**  — Epoch-level metrics (loss, accuracy, timing)
- **health.log** — Warnings, anomalies, and health-check results
- **debug.log**  — Full EXPERT-level detail (layer stats, gradients, weights)
                   Only written when actually needed.

Each file uses newline-delimited JSON (NDJSON) for easy streaming,
grep-ability, and incremental writes.

Usage::

    from neurogebra.logging.tiered_storage import TieredStorage

    storage = TieredStorage(base_dir="./training_logs")
    logger = TrainingLogger(level=LogLevel.EXPERT)
    logger.add_backend(storage)

    # … run training …
    storage.flush()
    storage.close()
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

from neurogebra.logging.logger import LogEvent, LogLevel


# ---------------------------------------------------------------------------
# Tier classification
# ---------------------------------------------------------------------------

# Events that go to basic.log
_BASIC_EVENTS = frozenset({
    "train_start", "train_end",
    "epoch_start", "epoch_end",
})

# Events that go to health.log
_HEALTH_EVENTS = frozenset({
    "health_check",
})

# Severity levels that always route to health.log regardless of event type
_HEALTH_SEVERITIES = frozenset({"warning", "danger", "critical"})

# Everything else goes to debug.log (layer_forward, layer_backward, etc.)


class TieredStorage:
    """
    Backend for :class:`TrainingLogger` that writes events into three
    separate NDJSON files based on their tier.

    Attributes:
        basic_path: Path to ``basic.log``.
        health_path: Path to ``health.log``.
        debug_path: Path to ``debug.log``.
    """

    def __init__(
        self,
        base_dir: str = "./training_logs",
        basic_filename: str = "basic.log",
        health_filename: str = "health.log",
        debug_filename: str = "debug.log",
        write_debug: bool = True,
        buffer_size: int = 50,
    ):
        """
        Args:
            base_dir: Directory for log files.
            basic_filename: Name of the epoch-metrics log file.
            health_filename: Name of the health/warnings log file.
            debug_filename: Name of the debug-level log file.
            write_debug: Whether to write debug-tier events at all.
                         Set to ``False`` in production to save I/O.
            buffer_size: Number of events to buffer before flushing to disk.
        """
        self.base_dir = base_dir
        self.basic_path = os.path.join(base_dir, basic_filename)
        self.health_path = os.path.join(base_dir, health_filename)
        self.debug_path = os.path.join(base_dir, debug_filename)
        self.write_debug = write_debug

        self._buffer_size = buffer_size
        self._basic_buffer: List[str] = []
        self._health_buffer: List[str] = []
        self._debug_buffer: List[str] = []

        self._opened = False
        self._basic_fh = None
        self._health_fh = None
        self._debug_fh = None

        # Stats
        self.basic_count = 0
        self.health_count = 0
        self.debug_count = 0

    # ------------------------------------------------------------------
    # Backend interface (called by TrainingLogger._emit)
    # ------------------------------------------------------------------

    def handle_event(self, event: LogEvent) -> None:
        """Classify and route the event to the appropriate tier."""
        record = self._serialise(event)
        line = json.dumps(record, default=str)

        tier = self._classify(event)

        if tier == "basic":
            self._basic_buffer.append(line)
            self.basic_count += 1
            if len(self._basic_buffer) >= self._buffer_size:
                self._flush_buffer("basic")

        elif tier == "health":
            self._health_buffer.append(line)
            self.health_count += 1
            if len(self._health_buffer) >= self._buffer_size:
                self._flush_buffer("health")

        else:  # debug
            if self.write_debug:
                self._debug_buffer.append(line)
                self.debug_count += 1
                if len(self._debug_buffer) >= self._buffer_size:
                    self._flush_buffer("debug")

    # ------------------------------------------------------------------
    # Specific event handlers for named dispatch
    # ------------------------------------------------------------------

    def handle_train_start(self, event: LogEvent) -> None:
        self.handle_event(event)

    def handle_train_end(self, event: LogEvent) -> None:
        self.handle_event(event)
        self.flush()

    def handle_epoch_start(self, event: LogEvent) -> None:
        self.handle_event(event)

    def handle_epoch_end(self, event: LogEvent) -> None:
        self.handle_event(event)
        # Flush basic tier at end of each epoch
        self._flush_buffer("basic")

    def handle_health_check(self, event: LogEvent) -> None:
        self.handle_event(event)
        # Health events are flushed immediately (important)
        self._flush_buffer("health")

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    @staticmethod
    def _classify(event: LogEvent) -> str:
        """Return 'basic', 'health', or 'debug'."""
        if event.event_type in _HEALTH_EVENTS:
            return "health"
        if event.severity in _HEALTH_SEVERITIES:
            return "health"
        if event.event_type in _BASIC_EVENTS:
            return "basic"
        return "debug"

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    @staticmethod
    def _serialise(event: LogEvent) -> Dict[str, Any]:
        return {
            "event_type": event.event_type,
            "level": event.level.name,
            "timestamp": event.timestamp,
            "epoch": event.epoch,
            "batch": event.batch,
            "layer_name": event.layer_name,
            "layer_index": event.layer_index,
            "severity": event.severity,
            "message": event.message,
            "data": _safe(event.data),
        }

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------

    def _ensure_dir(self) -> None:
        if not self._opened:
            os.makedirs(self.base_dir, exist_ok=True)
            self._opened = True

    def _flush_buffer(self, tier: str) -> None:
        buf_attr = f"_{tier}_buffer"
        path_attr = f"{tier}_path"
        buf: List[str] = getattr(self, buf_attr)
        if not buf:
            return
        self._ensure_dir()
        path = getattr(self, path_attr)
        with open(path, "a", encoding="utf-8") as f:
            for line in buf:
                f.write(line + "\n")
        buf.clear()

    def flush(self) -> None:
        """Flush all buffered events to disk."""
        self._flush_buffer("basic")
        self._flush_buffer("health")
        if self.write_debug:
            self._flush_buffer("debug")

    def close(self) -> None:
        """Flush and release resources."""
        self.flush()

    # ------------------------------------------------------------------
    # Reading helpers
    # ------------------------------------------------------------------

    def read_basic(self) -> List[Dict[str, Any]]:
        """Read all basic-tier events from disk."""
        return self._read_ndjson(self.basic_path)

    def read_health(self) -> List[Dict[str, Any]]:
        """Read all health-tier events from disk."""
        return self._read_ndjson(self.health_path)

    def read_debug(self) -> List[Dict[str, Any]]:
        """Read all debug-tier events from disk."""
        return self._read_ndjson(self.debug_path)

    @staticmethod
    def _read_ndjson(path: str) -> List[Dict[str, Any]]:
        if not os.path.exists(path):
            return []
        events = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return events

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Return file-size and event-count statistics."""
        def _size(path):
            try:
                return os.path.getsize(path)
            except OSError:
                return 0

        return {
            "basic": {"events": self.basic_count, "size_bytes": _size(self.basic_path)},
            "health": {"events": self.health_count, "size_bytes": _size(self.health_path)},
            "debug": {"events": self.debug_count, "size_bytes": _size(self.debug_path)},
            "total_events": self.basic_count + self.health_count + self.debug_count,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe(obj: Any) -> Any:
    """Make an object JSON-serialisable."""
    if isinstance(obj, dict):
        return {k: _safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe(v) for v in obj]
    if isinstance(obj, float):
        if obj != obj:
            return "NaN"
        if obj == float("inf"):
            return "Infinity"
        if obj == float("-inf"):
            return "-Infinity"
        return obj
    if isinstance(obj, (int, str, bool, type(None))):
        return obj
    try:
        import numpy as np
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    return str(obj)
