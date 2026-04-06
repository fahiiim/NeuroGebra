"""
JAX bridge for Neurogebra expressions.

Converts Neurogebra expressions to JAX-compatible functions.
"""

from typing import Callable, Optional
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


def to_jax(expression: Expression) -> Callable:
    """
    Convert a Neurogebra Expression to a JAX-compatible function.

    Args:
        expression: Neurogebra Expression instance

    Returns:
        JAX-compatible function for eager evaluation.

    Raises:
        ImportError: If JAX is not installed
        ValueError: If expression has more than one runtime input variable

    Notes:
        This bridge uses NumPy-backed evaluation under the hood. It is intended
        for interoperability, not traced JIT/grad execution.
    """
    check_jax()

    runtime_vars = [
        str(var)
        for var in expression.variables
        if str(var) not in expression.params
    ]
    if len(runtime_vars) > 1:
        raise ValueError(
            "to_jax currently supports single-input expressions. "
            f"Found unresolved variables: {runtime_vars}"
        )
    input_var = runtime_vars[0] if runtime_vars else "x"

    def jax_fn(x):
        x_np = np.asarray(x)
        result = expression.eval(**{input_var: x_np})
        return jnp.array(result)

    return jax_fn


def to_jax_grad(expression: Expression, var: str = "x") -> Callable:
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
