"""
Neurogebra: Neural-powered mathematics for AI developers.

A unified library bridging symbolic mathematics, numerical computation,
and deep learning with educational features built-in.

v2.8.3 -- Python 3.9 Compatibility Hardening, Rich LaTeX Expression Rendering,
Hardware Acceleration Engine (C-JIT AVX2 & CUDA GPU), Physics-Informed Neural Networks (PINN),
Learning Rate Schedulers, Evaluation Metrics, Advanced Layers (LayerNorm, GAP2D, ResidualDense),
and 20 Production-Grade Executed Notebook Projects.
"""

__version__ = "2.8.3"
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

# Evaluation Metrics (v2.8.0)
from neurogebra.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    explained_variance_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    root_mean_squared_error,
)

# Optimization & Schedulers (v2.8.0)
from neurogebra.optim import (
    SGD,
    Adam,
    CosineAnnealingLR,
    ExponentialLR,
    LRScheduler,
    Optimizer,
    ReduceLROnPlateau,
    RMSprop,
    StepLR,
)

# Scientific Machine Learning & PINN (v2.8.0)
from neurogebra.science import PINN

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
    # Optimization & Schedulers (v2.8.0)
    "Optimizer",
    "SGD",
    "Adam",
    "RMSprop",
    "LRScheduler",
    "StepLR",
    "CosineAnnealingLR",
    "ExponentialLR",
    "ReduceLROnPlateau",
    # Evaluation Metrics (v2.8.0)
    "accuracy_score",
    "confusion_matrix",
    "precision_score",
    "recall_score",
    "f1_score",
    "classification_report",
    "mean_squared_error",
    "root_mean_squared_error",
    "mean_absolute_error",
    "r2_score",
    "explained_variance_score",
    # Science (v2.8.0)
    "PINN",
    # Meta
    "__version__",
]
