"""
TensorFlow bridge for Neurogebra expressions.

Converts Neurogebra expressions to TensorFlow-compatible functions.
"""

from typing import Any, Callable, Optional
import numpy as np

try:
    import tensorflow as tf

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from neurogebra.core.expression import Expression


def check_tensorflow():
    """Raise if TensorFlow is not installed."""
    if not TF_AVAILABLE:
        raise ImportError(
            "TensorFlow is required for this feature. "
            "Install it with: pip install neurogebra[frameworks] "
            "or pip install tensorflow"
        )


def to_tensorflow(expression: Expression) -> Callable:
    """
    Convert a Neurogebra Expression to a TensorFlow function.

    Args:
        expression: Neurogebra Expression instance

    Returns:
        TensorFlow-compatible function

    Raises:
        ImportError: If TensorFlow is not installed
        ValueError: If expression has more than one runtime input variable
    """
    check_tensorflow()

    runtime_vars = [
        str(var)
        for var in expression.variables
        if str(var) not in expression.params
    ]
    if len(runtime_vars) > 1:
        raise ValueError(
            "to_tensorflow currently supports single-input expressions. "
            f"Found unresolved variables: {runtime_vars}"
        )
    input_var = runtime_vars[0] if runtime_vars else "x"

    def _evaluate_numpy(x_np: np.ndarray) -> np.ndarray:
        result = expression.eval(**{input_var: x_np})
        return np.asarray(result, dtype=x_np.dtype)

    def tf_expression(x: "tf.Tensor") -> "tf.Tensor":
        output = tf.numpy_function(_evaluate_numpy, [x], Tout=x.dtype)
        output.set_shape(x.shape)
        return output

    return tf_expression


def to_keras_layer(expression: Expression, name: Optional[str] = None) -> Any:
    """
    Convert a Neurogebra Expression to a Keras Layer.

    Args:
        expression: Neurogebra Expression instance
        name: Optional layer name

    Returns:
        Keras Lambda layer

    Raises:
        ValueError: If expression has more than one runtime input variable
    """
    check_tensorflow()

    layer_name = name or f"neurogebra_{expression.name}"

    runtime_vars = [
        str(var)
        for var in expression.variables
        if str(var) not in expression.params
    ]
    if len(runtime_vars) > 1:
        raise ValueError(
            "to_keras_layer currently supports single-input expressions. "
            f"Found unresolved variables: {runtime_vars}"
        )
    input_var = runtime_vars[0] if runtime_vars else "x"

    def _evaluate_numpy(x_np: np.ndarray) -> np.ndarray:
        result = expression.eval(**{input_var: x_np})
        return np.asarray(result, dtype=np.float32)

    def layer_fn(x):
        output = tf.numpy_function(_evaluate_numpy, [x], tf.float32)
        output.set_shape(x.shape)
        return output

    return tf.keras.layers.Lambda(layer_fn, name=layer_name)
