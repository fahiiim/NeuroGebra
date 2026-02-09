"""
Neurogebra: Neural-powered mathematics for AI developers.

A unified library bridging symbolic mathematics, numerical computation,
and deep learning with educational features built-in.
"""

__version__ = "0.3.0"
__author__ = "Fahim Sarker"
__email__ = "fahimsarker0805@gmail.com"

from neurogebra.core.forge import MathForge
from neurogebra.core.expression import Expression
from neurogebra.core.neurocraft import NeuroCraft
from neurogebra.builders.model_builder import ModelBuilder, Model, Layer

__all__ = [
    "MathForge",
    "Expression",
    "NeuroCraft",
    "ModelBuilder",
    "Model",
    "Layer",
    "__version__",
]
