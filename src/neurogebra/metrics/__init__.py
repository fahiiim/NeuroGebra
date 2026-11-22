"""
Evaluation metrics module for NeuroGebra.

Includes classification and regression metrics with zero third-party dependencies.
"""

from neurogebra.metrics.classification import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from neurogebra.metrics.regression import (
    explained_variance_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)

__all__ = [
    # Classification
    "accuracy_score",
    "confusion_matrix",
    "precision_score",
    "recall_score",
    "f1_score",
    "classification_report",
    # Regression
    "mean_squared_error",
    "root_mean_squared_error",
    "mean_absolute_error",
    "mean_absolute_percentage_error",
    "r2_score",
    "explained_variance_score",
]
