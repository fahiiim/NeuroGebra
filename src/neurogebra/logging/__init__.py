"""
Neurogebra Training Observatory — Advanced Training Visualization & Logging.

A comprehensive, colorful training visualization system that exposes
the complete computational graph, layer-by-layer mathematics, gradient flow,
and intelligent health monitoring in real-time.

Modules:
    logger          — Core TrainingLogger with event system
    config          — LogConfig presets and customization
    monitors        — Gradient, Weight, Activation, Performance monitors
    health_checks   — Smart health diagnostics with actionable recommendations
    terminal_display— Rich terminal dashboard with ANSI colors
    formula_renderer— Mathematical formula pretty-printing
    image_logger    — Pixel-level image & activation map visualization
    exporters       — JSON, CSV, TensorBoard, WandB, HTML, Markdown export
    computation_graph — Full computational graph tracking
"""

from neurogebra.logging.logger import TrainingLogger, LogLevel, LogEvent
from neurogebra.logging.config import LogConfig
from neurogebra.logging.monitors import (
    GradientMonitor,
    WeightMonitor,
    ActivationMonitor,
    PerformanceMonitor,
)
from neurogebra.logging.health_checks import SmartHealthChecker
from neurogebra.logging.terminal_display import TerminalDisplay
from neurogebra.logging.formula_renderer import FormulaRenderer
from neurogebra.logging.image_logger import ImageLogger
from neurogebra.logging.exporters import (
    JSONExporter,
    CSVExporter,
    HTMLExporter,
    MarkdownExporter,
)

__all__ = [
    "TrainingLogger",
    "LogLevel",
    "LogEvent",
    "LogConfig",
    "GradientMonitor",
    "WeightMonitor",
    "ActivationMonitor",
    "PerformanceMonitor",
    "SmartHealthChecker",
    "TerminalDisplay",
    "FormulaRenderer",
    "ImageLogger",
    "JSONExporter",
    "CSVExporter",
    "HTMLExporter",
    "MarkdownExporter",
]
