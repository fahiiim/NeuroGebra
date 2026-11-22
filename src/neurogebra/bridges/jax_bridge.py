"""
JAX bridge for Neurogebra expressions.

Converts Neurogebra expressions to JAX-compatible functions.
"""

from typing import Callable

import numpy as np
import sympy as sp

try:
    import jax
    import jax.numpy as jnp

    JAX_AVAILABLE = True
    _ = jax
except ImportError:
    JAX_AVAILABLE = False
    jax = None  # type: ignore

from neurogebra.core.expression import Expression


def check_jax():
    """Raise if JAX is not installed."""
    if not JAX_AVAILABLE:
        raise ImportError(
            "JAX is required for this feature. Install it with: pip install jax jaxlib"
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

    if hasattr(expression, "layers"):
        raise TypeError(
            "to_jax converts mathematical Expression objects to JAX functions. "
            "Model conversion to JAX/Flax neural modules is not currently supported; "
            "use to_pytorch or to_tensorflow for neural network models."
        )

    runtime_vars = [
        str(var) for var in expression.variables if str(var) not in expression.params
    ]
    if len(runtime_vars) > 1:
        raise ValueError(
            "to_jax currently supports single-input expressions. "
            f"Found unresolved variables: {runtime_vars}"
        )
    input_var = runtime_vars[0] if runtime_vars else "x"

    try:
        sym_expr = expression.symbolic_expr.subs(expression.params)
        var_symbol = sp.Symbol(input_var)
        return sp.lambdify(var_symbol, sym_expr, modules=["jax", "jax.numpy"])
    except Exception:

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
