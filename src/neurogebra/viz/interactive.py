"""
Interactive visualization tools for Neurogebra.

Provides interactive plotting using plotly (optional dependency).
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from neurogebra.core.expression import Expression

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def check_plotly():
    """Raise if plotly is not installed."""
    if not PLOTLY_AVAILABLE:
        raise ImportError(
            "Plotly is required for interactive visualizations. "
            "Install it with: pip install neurogebra[viz] "
            "or pip install plotly"
        )


def interactive_plot(
    expression: Expression,
    x_range: Tuple[float, float] = (-5, 5),
    n_points: int = 500,
    title: Optional[str] = None,
) -> "go.Figure":
    """
    Create an interactive plot of an expression using Plotly.

    Args:
        expression: Expression to plot
        x_range: (min, max) range for x values
        n_points: Number of evaluation points
        title: Plot title

    Returns:
        Plotly Figure object
    """
    check_plotly()

    x = np.linspace(x_range[0], x_range[1], n_points)
    y = np.array([expression.eval(x=xi) for xi in x])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            name=expression.name,
            line=dict(width=2),
            hovertemplate="x: %{x:.3f}<br>f(x): %{y:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=title or f"{expression.name}: {expression.formula}",
        xaxis_title="x",
        yaxis_title="f(x)",
        template="plotly_white",
        hovermode="x unified",
    )

    return fig


def interactive_comparison(
    expressions: List[Expression],
    x_range: Tuple[float, float] = (-5, 5),
    n_points: int = 500,
    title: str = "Expression Comparison",
) -> "go.Figure":
    """
    Interactive comparison of multiple expressions.

    Args:
        expressions: List of Expressions to compare
        x_range: (min, max) range
        n_points: Number of points
        title: Plot title

    Returns:
        Plotly Figure object
    """
    check_plotly()

    fig = go.Figure()
    x = np.linspace(x_range[0], x_range[1], n_points)

    for expr in expressions:
        try:
            y = np.array([expr.eval(x=xi) for xi in x])
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    name=expr.name,
                    line=dict(width=2),
                )
            )
        except Exception as e:
            print(f"Warning: Could not plot {expr.name}: {e}")

    fig.update_layout(
        title=title,
        xaxis_title="x",
        yaxis_title="f(x)",
        template="plotly_white",
        hovermode="x unified",
    )

    return fig
