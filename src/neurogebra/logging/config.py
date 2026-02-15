"""
LogConfig — Configuration presets for the Training Observatory.

Provides named presets (``minimal``, ``standard``, ``verbose``, ``research``,
``production``) and a builder API for composing custom configurations.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from neurogebra.logging.logger import LogLevel


@dataclass
class LogConfig:
    """
    Configuration for the Training Observatory.

    Attributes:
        level: Verbosity level (SILENT → DEBUG).
        show_formulas: Display mathematical formulas during forward/backward.
        show_gradients: Show per-layer gradient statistics.
        show_weights: Show weight distribution information.
        show_activations: Show activation statistics.
        show_timing: Show per-layer and per-batch timing.
        show_health_checks: Enable intelligent health diagnostics.
        show_images: Enable pixel-level image / activation-map rendering.
        color_scheme: Mapping of severity → colour name for *rich*.
        export_formats: List of export backends to activate (json, csv, html …).
        export_dir: Directory for exported files.
        health_check_interval: Run health checks every N epochs.
        max_image_width: Maximum terminal width for ASCII image rendering.
        sparkline_length: Length of inline sparkline charts.
    """

    level: LogLevel = LogLevel.BASIC
    show_formulas: bool = False
    show_gradients: bool = False
    show_weights: bool = False
    show_activations: bool = False
    show_timing: bool = False
    show_health_checks: bool = True
    show_images: bool = False

    color_scheme: Dict[str, str] = field(default_factory=lambda: {
        "info": "blue",
        "success": "green",
        "warning": "yellow",
        "danger": "bold red",
        "header": "bold cyan",
        "muted": "dim white",
        "formula": "magenta",
        "layer": "bright_cyan",
    })

    export_formats: List[str] = field(default_factory=list)
    export_dir: str = "./training_logs"
    health_check_interval: int = 1
    max_image_width: int = 60
    sparkline_length: int = 20
    layout_mode: str = "compact"  # compact | detailed | expert | debug

    # ------------------------------------------------------------------
    # Named presets
    # ------------------------------------------------------------------

    @classmethod
    def minimal(cls) -> "LogConfig":
        """Epoch-level progress only."""
        return cls(level=LogLevel.BASIC, layout_mode="compact")

    @classmethod
    def standard(cls) -> "LogConfig":
        """Layer info + health checks + timing."""
        return cls(
            level=LogLevel.DETAILED,
            show_timing=True,
            show_health_checks=True,
            layout_mode="detailed",
        )

    @classmethod
    def verbose(cls) -> "LogConfig":
        """Full layer-by-layer breakdown with formulas + gradients."""
        return cls(
            level=LogLevel.EXPERT,
            show_formulas=True,
            show_gradients=True,
            show_weights=True,
            show_activations=True,
            show_timing=True,
            show_health_checks=True,
            show_images=True,
            layout_mode="expert",
        )

    @classmethod
    def research(cls) -> "LogConfig":
        """Everything + JSON + HTML export."""
        cfg = cls.verbose()
        cfg.export_formats = ["json", "csv", "html"]
        return cfg

    @classmethod
    def production(cls) -> "LogConfig":
        """Silent console, export to JSON only."""
        return cls(
            level=LogLevel.SILENT,
            export_formats=["json"],
            layout_mode="compact",
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level.name,
            "show_formulas": self.show_formulas,
            "show_gradients": self.show_gradients,
            "show_weights": self.show_weights,
            "show_activations": self.show_activations,
            "show_timing": self.show_timing,
            "show_health_checks": self.show_health_checks,
            "show_images": self.show_images,
            "color_scheme": self.color_scheme,
            "export_formats": self.export_formats,
            "export_dir": self.export_dir,
            "health_check_interval": self.health_check_interval,
            "max_image_width": self.max_image_width,
            "sparkline_length": self.sparkline_length,
            "layout_mode": self.layout_mode,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LogConfig":
        d = dict(d)
        if "level" in d and isinstance(d["level"], str):
            d["level"] = LogLevel[d["level"]]
        return cls(**d)

    def save(self, path: str) -> None:
        """Persist config as JSON."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "LogConfig":
        """Load config from JSON."""
        with open(path) as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def from_env(cls) -> "LogConfig":
        """Build config from ``NEUROGEBRA_LOG_*`` environment variables."""
        level_name = os.environ.get("NEUROGEBRA_LOG_LEVEL", "BASIC").upper()
        level = LogLevel[level_name] if level_name in LogLevel.__members__ else LogLevel.BASIC
        layout = os.environ.get("NEUROGEBRA_LOG_LAYOUT", "compact")
        return cls(level=level, layout_mode=layout)
