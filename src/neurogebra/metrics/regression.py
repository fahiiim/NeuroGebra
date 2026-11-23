"""
Regression metrics for model evaluation in NeuroGebra.
"""

from __future__ import annotations

import numpy as np


def _check_reg_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    if yt.ndim == 1 and yp.ndim == 2 and yp.shape[1] == 1:
        yp = yp.squeeze(1)
    elif yp.ndim == 1 and yt.ndim == 2 and yt.shape[1] == 1:
        yt = yt.squeeze(1)
    if yt.shape != yp.shape:
        raise ValueError(
            f"Shape mismatch in regression metric: y_true {yt.shape} vs y_pred {yp.shape}"
        )
    return yt, yp


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error regression loss."""
    yt, yp = _check_reg_arrays(y_true, y_pred)
    return float(np.mean((yt - yp) ** 2))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error regression loss."""
    yt, yp = _check_reg_arrays(y_true, y_pred)
    return float(np.mean(np.abs(yt - yp)))


def mean_absolute_percentage_error(
    y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8
) -> float:
    """Mean absolute percentage error (MAPE)."""
    yt, yp = _check_reg_arrays(y_true, y_pred)
    denom = np.maximum(np.abs(yt), eps)
    return float(np.mean(np.abs((yt - yp) / denom))) * 100.0


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    R^2 (coefficient of determination) regression score function.

    Formula:
        1 - (SS_res / SS_tot)
    """
    yt, yp = _check_reg_arrays(y_true, y_pred)
    ss_res = np.sum((yt - yp) ** 2)
    mean_y = np.mean(yt)
    ss_tot = np.sum((yt - mean_y) ** 2)
    if ss_tot == 0:
        return 0.0 if ss_res > 0 else 1.0
    return float(1.0 - (ss_res / ss_tot))


def explained_variance_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Explained variance regression score function.

    Formula:
        1 - Var[y - y_hat] / Var[y]
    """
    yt, yp = _check_reg_arrays(y_true, y_pred)
    diff = yt - yp
    var_diff = np.var(diff)
    var_y = np.var(yt)
    if var_y == 0:
        return 0.0 if var_diff > 0 else 1.0
    return float(1.0 - (var_diff / var_y))
