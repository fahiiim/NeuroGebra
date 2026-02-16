"""
Helper utilities for Neurogebra.

Provides common utility functions used across the library.
"""

from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np


def validate_array(data: Any, name: str = "input") -> np.ndarray:
    """
    Validate and convert input to numpy array.

    Args:
        data: Input data (list, tuple, numpy array, or scalar)
        name: Name for error messages

    Returns:
        NumPy array

    Raises:
        TypeError: If data cannot be converted
    """
    try:
        arr = np.asarray(data, dtype=np.float64)
        return arr
    except (ValueError, TypeError) as e:
        raise TypeError(f"Cannot convert {name} to numpy array: {e}")


def clip_gradients(
    gradients: Dict[str, float],
    max_norm: float = 1.0,
) -> Dict[str, float]:
    """
    Clip gradients by global norm.

    Args:
        gradients: Dictionary of parameter name -> gradient value
        max_norm: Maximum allowed gradient norm

    Returns:
        Clipped gradients
    """
    total_norm = np.sqrt(sum(g**2 for g in gradients.values()))

    if total_norm > max_norm:
        scale = max_norm / total_norm
        return {k: v * scale for k, v in gradients.items()}

    return gradients


def numerical_gradient(
    func: Callable, x: np.ndarray, epsilon: float = 1e-5
) -> np.ndarray:
    """
    Compute numerical gradient using central differences.

    Args:
        func: Function to differentiate
        x: Point at which to compute gradient
        epsilon: Step size for finite differences

    Returns:
        Gradient array
    """
    x = np.asarray(x, dtype=np.float64)
    grad = np.zeros_like(x)

    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += epsilon
        x_minus[i] -= epsilon

        grad[i] = (func(x_plus) - func(x_minus)) / (2 * epsilon)

    return grad


def normalize(data: np.ndarray, method: str = "minmax") -> np.ndarray:
    """
    Normalize data.

    Args:
        data: Input array
        method: Normalization method ('minmax', 'standard', 'l2')

    Returns:
        Normalized array
    """
    data = np.asarray(data, dtype=np.float64)

    if method == "minmax":
        data_min = data.min()
        data_max = data.max()
        if data_max - data_min == 0:
            return np.zeros_like(data)
        return (data - data_min) / (data_max - data_min)

    elif method == "standard":
        mean = data.mean()
        std = data.std()
        if std == 0:
            return np.zeros_like(data)
        return (data - mean) / std

    elif method == "l2":
        norm = np.linalg.norm(data)
        if norm == 0:
            return np.zeros_like(data)
        return data / norm

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def generate_data(
    func: Callable,
    x_range: tuple = (-5, 5),
    n_points: int = 100,
    noise_std: float = 0.0,
    seed: Optional[int] = None,
) -> tuple:
    """
    Generate synthetic data from a function.

    Args:
        func: Function to generate data from
        x_range: (min, max) range for x values
        n_points: Number of data points
        noise_std: Standard deviation of Gaussian noise
        seed: Random seed for reproducibility

    Returns:
        Tuple of (X, y) arrays
    """
    if seed is not None:
        np.random.seed(seed)

    X = np.linspace(x_range[0], x_range[1], n_points)
    y = np.array([func(x) for x in X], dtype=np.float64)

    if noise_std > 0:
        y += np.random.normal(0, noise_std, size=n_points)

    return X, y
