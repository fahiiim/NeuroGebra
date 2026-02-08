"""
TensorFlow bridge for Neurogebra expressions.

Converts Neurogebra expressions to TensorFlow-compatible functions.
"""

from typing import Optional
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


def to_tensorflow(expression: Expression):
    """
    Convert a Neurogebra Expression to a TensorFlow function.

    Args:
        expression: Neurogebra Expression instance

    Returns:
        TensorFlow-compatible function

    Raises:
        ImportError: If TensorFlow is not installed
    """
    check_tensorflow()

    @tf.function
    def tf_expression(x):
        x_np = x.numpy()
        result = np.vectorize(
            lambda val: float(expression.symbolic_expr.subs("x", val))
        )(x_np)
        return tf.constant(result, dtype=x.dtype)

    return tf_expression


def to_keras_layer(expression: Expression, name: Optional[str] = None):
    """
    Convert a Neurogebra Expression to a Keras Layer.

    Args:
        expression: Neurogebra Expression instance
        name: Optional layer name

    Returns:
        Keras Lambda layer
    """
    check_tensorflow()

    layer_name = name or f"neurogebra_{expression.name}"

    def layer_fn(x):
        # Use numpy callback for evaluation
        return tf.numpy_function(
            lambda x_np: np.vectorize(
                lambda val: float(expression.symbolic_expr.subs("x", val))
            )(x_np).astype(np.float32),
            [x],
            tf.float32,
        )

    return tf.keras.layers.Lambda(layer_fn, name=layer_name)
