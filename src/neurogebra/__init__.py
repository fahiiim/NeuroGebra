"""
Neurogebra: Neural-powered mathematics for AI developers.

A unified library bridging symbolic mathematics, numerical computation,
and deep learning with educational features built-in.

v1.3.0 — Observatory Pro: adaptive logging, automated health warnings,
epoch summarization, tiered storage, visual dashboards, and training
fingerprinting for full reproducibility.
"""

__version__ = "1.3.0"
__author__ = "Fahim Sarker"
__email__ = "fahimsarker0805@gmail.com"

from neurogebra.core.forge import MathForge
from neurogebra.core.expression import Expression
from neurogebra.core.neurocraft import NeuroCraft
from neurogebra.builders.model_builder import ModelBuilder, Model, Layer

# Training Observatory (v1.2.1+)
from neurogebra.logging.logger import TrainingLogger, LogLevel, LogEvent
from neurogebra.logging.config import LogConfig

# Observatory Pro (v1.3.0)
from neurogebra.logging.adaptive import AdaptiveLogger, AnomalyConfig
from neurogebra.logging.health_warnings import AutoHealthWarnings, WarningConfig
from neurogebra.logging.epoch_summary import EpochSummarizer
from neurogebra.logging.tiered_storage import TieredStorage
from neurogebra.logging.dashboard import DashboardExporter
from neurogebra.logging.fingerprint import TrainingFingerprint

__all__ = [
    # Core
    "MathForge",
    "Expression",
    "NeuroCraft",
    # Builders
    "ModelBuilder",
    "Model",
    "Layer",
    # Training Observatory
    "TrainingLogger",
    "LogLevel",
    "LogEvent",
    "LogConfig",
    # Observatory Pro (v1.3.0)
    "AdaptiveLogger",
    "AnomalyConfig",
    "AutoHealthWarnings",
    "WarningConfig",
    "EpochSummarizer",
    "TieredStorage",
    "DashboardExporter",
    "TrainingFingerprint",
    # Meta
    "__version__",
]
