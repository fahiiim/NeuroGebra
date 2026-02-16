"""
Plotting utilities for Neurogebra.

Provides static plotting functions for expressions using matplotlib.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
from neurogebra.core.expression import Expression


def plot_expression(
    expression: Expression,
    x_range: Tuple[float, float] = (-5, 5),
    n_points: int = 500,
    title: Optional[str] = None,
    xlabel: str = "x",
    ylabel: str = "f(x)",
    figsize: Tuple[int, int] = (8, 5),
    show_grid: bool = True,
    show_formula: bool = True,
    ax: Optional[plt.Axes] = None,
    **eval_kwargs: Any,
) -> plt.Figure:
    """
    Plot a single expression.

    Args:
        expression: Expression to plot
        x_range: (min, max) range for x-axis
        n_points: Number of points to evaluate
        title: Plot title (defaults to expression name)
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size
        show_grid: Whether to show grid
        show_formula: Whether to show formula in legend
        ax: Optional matplotlib Axes to plot on
        **eval_kwargs: Additional keyword arguments for eval

    Returns:
        matplotlib Figure
    """
    x = np.linspace(x_range[0], x_range[1], n_points)

    # Evaluate expression
    try:
        y = np.array([expression.eval(x=xi, **eval_kwargs) for xi in x])
    except Exception:
        y = np.vectorize(lambda xi: expression.eval(x=xi, **eval_kwargs))(x)

    # Create figure if no axes provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    label = f"${expression.formula}$" if show_formula else expression.name
    ax.plot(x, y, label=label, linewidth=2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title or f"{expression.name}")
    ax.legend(fontsize=9)

    if show_grid:
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color="k", linewidth=0.5)
        ax.axvline(x=0, color="k", linewidth=0.5)

    fig.tight_layout()
    return fig


def plot_comparison(
    expressions: List[Expression],
    x_range: Tuple[float, float] = (-5, 5),
    n_points: int = 500,
    title: str = "Expression Comparison",
    figsize: Tuple[int, int] = (10, 6),
    show_grid: bool = True,
) -> plt.Figure:
    """
    Plot multiple expressions on the same axes for comparison.

    Args:
        expressions: List of Expressions to compare
        x_range: (min, max) range for x-axis
        n_points: Number of points
        title: Plot title
        figsize: Figure size
        show_grid: Whether to show grid

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    x = np.linspace(x_range[0], x_range[1], n_points)

    for expr in expressions:
        try:
            y = np.array([expr.eval(x=xi) for xi in x])
            ax.plot(x, y, label=expr.name, linewidth=2)
        except Exception as e:
            print(f"Warning: Could not plot {expr.name}: {e}")

    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_title(title)
    ax.legend(fontsize=9)

    if show_grid:
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color="k", linewidth=0.5)
        ax.axvline(x=0, color="k", linewidth=0.5)

    fig.tight_layout()
    return fig


def plot_gradient(
    expression: Expression,
    var: str = "x",
    x_range: Tuple[float, float] = (-5, 5),
    n_points: int = 500,
    figsize: Tuple[int, int] = (10, 5),
) -> plt.Figure:
    """
    Plot expression alongside its gradient.

    Args:
        expression: Expression to plot
        var: Variable to differentiate with respect to
        x_range: (min, max) range for x-axis
        n_points: Number of points
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    grad_expr = expression.gradient(var)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    x = np.linspace(x_range[0], x_range[1], n_points)

    # Plot original
    y = np.array([expression.eval(x=xi) for xi in x])
    axes[0].plot(x, y, "b-", linewidth=2)
    axes[0].set_title(f"{expression.name}")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("f(x)")
    axes[0].grid(True, alpha=0.3)

    # Plot gradient
    dy = np.array([grad_expr.eval(x=xi) for xi in x])
    axes[1].plot(x, dy, "r-", linewidth=2)
    axes[1].set_title(f"d({expression.name})/d{var}")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("f'(x)")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_training_history(
    history: Dict[str, List],
    figsize: Tuple[int, int] = (12, 4),
) -> plt.Figure:
    """
    Plot training history (loss and accuracy).

    Args:
        history: Training history dict with 'loss', 'val_loss',
                 'accuracy', 'val_accuracy' keys
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    has_val = bool(history.get("val_loss"))
    has_acc = bool(history.get("accuracy"))
    n_plots = 1 + int(has_acc)

    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    if n_plots == 1:
        axes = [axes]

    epochs = range(1, len(history["loss"]) + 1)

    # Loss plot
    axes[0].plot(
        epochs, history["loss"], "b-", linewidth=2, label="Training Loss"
    )
    if has_val:
        axes[0].plot(
            epochs,
            history["val_loss"],
            "r--",
            linewidth=2,
            label="Validation Loss",
        )
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Progress")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy plot
    if has_acc:
        axes[1].plot(
            epochs,
            history["accuracy"],
            "b-",
            linewidth=2,
            label="Training Acc",
        )
        if history.get("val_accuracy"):
            axes[1].plot(
                epochs,
                history["val_accuracy"],
                "r--",
                linewidth=2,
                label="Validation Acc",
            )
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Model Accuracy")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    return fig
