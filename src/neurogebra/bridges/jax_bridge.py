"""
JAX bridge for Neurogebra expressions.

Converts Neurogebra expressions to JAX-compatible functions.
"""

from typing import Optional
import numpy as np

try:
    import jax
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

from neurogebra.core.expression import Expression


def check_jax():
    """Raise if JAX is not installed."""
    if not JAX_AVAILABLE:
        raise ImportError(
            "JAX is required for this feature. "
            "Install it with: pip install jax jaxlib"
        )


def to_jax(expression: Expression):
    """
    Convert a Neurogebra Expression to a JAX-compatible function.

    Args:
        expression: Neurogebra Expression instance

    Returns:
        JAX-compatible function that can be used with jit, grad, etc.

    Raises:
        ImportError: If JAX is not installed
    """
    check_jax()

    def jax_fn(x):
        # Convert to numpy, evaluate, convert back
        x_np = np.asarray(x)
        result = np.vectorize(
            lambda val: float(expression.symbolic_expr.subs("x", val))
        )(x_np)
        return jnp.array(result)

    return jax_fn


def to_jax_grad(expression: Expression, var: str = "x"):
    """
    Convert expression and return its gradient as a JAX function.

    Args:
        expression: Neurogebra Expression instance
        var: Variable to differentiate with respect to

    Returns:
        JAX function computing the gradient
    """
    check_jax()

    grad_expr = expression.gradient(var)
    return to_jax(grad_expr)
