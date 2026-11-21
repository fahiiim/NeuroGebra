"""
Neurogebra: Neural-powered mathematics for AI developers.

A unified library bridging symbolic mathematics, numerical computation,
and deep learning with educational features built-in.

v2.6.0 -- Hardware Acceleration Engine (C-JIT AVX2 & CUDA GPU), real CNN operations,
autograd overhaul, Observatory Pro adaptive diagnostics, and full training transparency.
"""

__version__ = "2.6.0"
__author__ = "Fahim Sarker"
__email__ = "fahimsarker0805@gmail.com"

# Acceleration Engine (v2.6.0+)
from neurogebra.acceleration import (
    compile_c,
    compile_cuda,
    get_acceleration_info,
    is_c_available,
    is_cuda_available,
)
from neurogebra.builders.model_builder import Layer, Model, ModelBuilder
from neurogebra.core.expression import Expression
from neurogebra.core.forge import MathForge
from neurogebra.core.neurocraft import NeuroCraft

# Observatory Pro (v1.3.0)
from neurogebra.logging.adaptive import AdaptiveLogger, AnomalyConfig
from neurogebra.logging.config import LogConfig
from neurogebra.logging.dashboard import DashboardExporter
from neurogebra.logging.epoch_summary import EpochSummarizer
from neurogebra.logging.fingerprint import TrainingFingerprint
from neurogebra.logging.health_warnings import AutoHealthWarnings, WarningConfig

# Training Observatory (v1.2.1+)
from neurogebra.logging.logger import LogEvent, LogLevel, TrainingLogger
from neurogebra.logging.tiered_storage import TieredStorage

# Acceleration Engine (v2.6.0+)
from neurogebra.acceleration import (
    compile_c,
    compile_cuda,
    get_acceleration_info,
    is_c_available,
    is_cuda_available,
)

__all__ = [
    # Core
    "MathForge",
    "Expression",
    "NeuroCraft",
    # Builders
    "ModelBuilder",
    "Model",
    "Layer",
    # Hardware Acceleration
    "compile_c",
    "compile_cuda",
    "get_acceleration_info",
    "is_c_available",
    "is_cuda_available",
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
