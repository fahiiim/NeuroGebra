"""
Neurogebra: Neural-powered mathematics for AI developers.

A unified library bridging symbolic mathematics, numerical computation,
and deep learning with educational features built-in.

v1.2.1 — Training Observatory: colourful, depth-level mathematical
logging and visualization during model training.
"""

__version__ = "1.2.1"
__author__ = "Fahim Sarker"
__email__ = "fahimsarker0805@gmail.com"

from neurogebra.core.forge import MathForge
from neurogebra.core.expression import Expression
from neurogebra.core.neurocraft import NeuroCraft
from neurogebra.builders.model_builder import ModelBuilder, Model, Layer

# Training Observatory (v1.2.1)
from neurogebra.logging.logger import TrainingLogger, LogLevel, LogEvent
from neurogebra.logging.config import LogConfig

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
    # Meta
    "__version__",
]
