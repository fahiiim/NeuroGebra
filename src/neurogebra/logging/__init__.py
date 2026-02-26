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
    health_warnings — Automated threshold-based health warning system
    adaptive        — Smart anomaly-triggered adaptive logging
    epoch_summary   — Per-epoch statistical summarization
    tiered_storage  — Tiered log file separation (basic/health/debug)
    dashboard       — Visual HTML dashboard + TensorBoard/WandB bridges
    fingerprint     — Training reproducibility fingerprint
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
from neurogebra.logging.health_warnings import AutoHealthWarnings, HealthWarning, WarningConfig
from neurogebra.logging.adaptive import AdaptiveLogger, AnomalyConfig, AnomalyRecord
from neurogebra.logging.epoch_summary import EpochSummarizer, EpochSummary, EpochStats
from neurogebra.logging.tiered_storage import TieredStorage
from neurogebra.logging.dashboard import DashboardExporter, TensorBoardBridge, WandBBridge
from neurogebra.logging.fingerprint import TrainingFingerprint
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
    # Core
    "TrainingLogger",
    "LogLevel",
    "LogEvent",
    "LogConfig",
    # Monitors
    "GradientMonitor",
    "WeightMonitor",
    "ActivationMonitor",
    "PerformanceMonitor",
    # Health
    "SmartHealthChecker",
    "AutoHealthWarnings",
    "HealthWarning",
    "WarningConfig",
    # Adaptive logging (v1.3.0)
    "AdaptiveLogger",
    "AnomalyConfig",
    "AnomalyRecord",
    # Epoch summarization (v1.3.0)
    "EpochSummarizer",
    "EpochSummary",
    "EpochStats",
    # Tiered storage (v1.3.0)
    "TieredStorage",
    # Dashboard & bridges (v1.3.0)
    "DashboardExporter",
    "TensorBoardBridge",
    "WandBBridge",
    # Fingerprint (v1.3.0)
    "TrainingFingerprint",
    # Display
    "TerminalDisplay",
    "FormulaRenderer",
    "ImageLogger",
    # Exporters
    "JSONExporter",
    "CSVExporter",
    "HTMLExporter",
    "MarkdownExporter",
]
