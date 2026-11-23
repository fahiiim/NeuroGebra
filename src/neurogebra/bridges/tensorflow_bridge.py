"""
TensorFlow bridge for Neurogebra expressions.

Converts Neurogebra expressions to TensorFlow-compatible functions.
"""

from __future__ import annotations

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


class KerasResidualDense(tf.keras.layers.Layer if TF_AVAILABLE else object):
    """TensorFlow/Keras implementation of NeuroGebra ResidualDense layer."""

    def __init__(
        self,
        units: int,
        activation: str = "relu",
        weights: Optional[np.ndarray] = None,
        bias: Optional[np.ndarray] = None,
        weights2: Optional[np.ndarray] = None,
        bias2: Optional[np.ndarray] = None,
        weights_skip: Optional[np.ndarray] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.units = units
        self.activation_name = (activation or "").lower()
        self._preset_w = weights
        self._preset_b = bias
        self._preset_w2 = weights2
        self._preset_b2 = bias2
        self._preset_wskip = weights_skip

    def build(self, input_shape):
        in_dim = int(input_shape[-1])
        w1_init = tf.constant_initializer(self._preset_w) if self._preset_w is not None else "glorot_uniform"
        b1_init = tf.constant_initializer(self._preset_b) if self._preset_b is not None else "zeros"
        self.w1 = self.add_weight("w1", shape=(in_dim, self.units), initializer=w1_init)
        self.b1 = self.add_weight("b1", shape=(self.units,), initializer=b1_init)

        w2_init = tf.constant_initializer(self._preset_w2) if self._preset_w2 is not None else "glorot_uniform"
        b2_init = tf.constant_initializer(self._preset_b2) if self._preset_b2 is not None else "zeros"
        self.w2 = self.add_weight("w2", shape=(self.units, self.units), initializer=w2_init)
        self.b2 = self.add_weight("b2", shape=(self.units,), initializer=b2_init)

        if self._preset_wskip is not None or in_dim != self.units:
            wskip_init = tf.constant_initializer(self._preset_wskip) if self._preset_wskip is not None else "glorot_uniform"
            self.w_skip = self.add_weight("w_skip", shape=(in_dim, self.units), initializer=wskip_init)
        else:
            self.w_skip = None
        super().build(input_shape)

    def call(self, inputs):
        h = tf.matmul(inputs, self.w1) + self.b1
        if self.activation_name == "relu":
            h = tf.nn.relu(h)
        elif self.activation_name == "sigmoid":
            h = tf.nn.sigmoid(h)
        elif self.activation_name == "tanh":
            h = tf.nn.tanh(h)
        elif self.activation_name == "gelu":
            h = tf.nn.gelu(h)
        out = tf.matmul(h, self.w2) + self.b2
        skip = tf.matmul(inputs, self.w_skip) if self.w_skip is not None else inputs
        return out + skip


def model_to_tensorflow(model):
    """Convert a Neurogebra Model to a TensorFlow/Keras Sequential model."""
    check_tensorflow()
    tf_layers = []
    for layer in model.layers:
        ltype = layer.layer_type.lower()
        if ltype == "dense":
            in_dim = layer.weights.shape[0] if layer.weights is not None else None
            keras_dense = tf.keras.layers.Dense(
                layer.units, activation=layer.activation
            )
            if in_dim is not None:
                keras_dense.build((None, in_dim))
                weights_to_set = [layer.weights]
                if layer.bias is not None:
                    weights_to_set.append(layer.bias)
                keras_dense.set_weights(weights_to_set)
            tf_layers.append(keras_dense)
        elif ltype == "dropout":
            tf_layers.append(tf.keras.layers.Dropout(layer.params.get("rate", 0.2)))
        elif ltype == "flatten":
            tf_layers.append(tf.keras.layers.Flatten())
        elif ltype == "batchnorm":
            tf_layers.append(tf.keras.layers.BatchNormalization())
        elif ltype == "conv2d":
            padding = str(layer.params.get("padding", "valid")).lower()
            k = layer.params.get("kernel_size", 3)
            c_out = layer.units or 32
            keras_conv = tf.keras.layers.Conv2D(
                c_out,
                kernel_size=k,
                padding=padding,
                activation=layer.activation,
            )
            if layer.weights is not None:
                in_c = layer.weights.shape[2]
                keras_conv.build((None, None, None, in_c))
                weights_to_set = [layer.weights]
                if layer.bias is not None:
                    weights_to_set.append(layer.bias)
                keras_conv.set_weights(weights_to_set)
            tf_layers.append(keras_conv)
        elif ltype == "maxpool2d":
            tf_layers.append(
                tf.keras.layers.MaxPooling2D(pool_size=layer.params.get("pool_size", 2))
            )
        elif ltype == "layernorm":
            eps = float(layer.params.get("eps", 1e-5))
            keras_ln = tf.keras.layers.LayerNormalization(epsilon=eps)
            if getattr(layer, "gamma", None) is not None and getattr(layer, "beta", None) is not None:
                keras_ln.build((None, len(layer.gamma)))
                keras_ln.set_weights([layer.gamma, layer.beta])
            tf_layers.append(keras_ln)
        elif ltype == "global_avgpool2d":
            tf_layers.append(tf.keras.layers.GlobalAveragePooling2D())
        elif ltype == "residual_dense":
            in_dim = layer.weights.shape[0] if layer.weights is not None else layer.units
            out_dim = layer.units or in_dim
            res_layer = KerasResidualDense(
                units=out_dim,
                activation=layer.activation or "relu",
                weights=layer.weights,
                bias=layer.bias,
                weights2=getattr(layer, "weights2", None),
                bias2=getattr(layer, "bias2", None),
                weights_skip=getattr(layer, "weights_skip", None),
            )
            if in_dim is not None:
                res_layer.build((None, in_dim))
            tf_layers.append(res_layer)
        else:
            raise NotImplementedError(
                f"TensorFlow bridge does not support layer type '{layer.layer_type}'"
            )
    return tf.keras.Sequential(tf_layers)


def to_tensorflow(obj: Any) -> Callable:
    """
    Convert a Neurogebra Expression or Model to a TensorFlow function or model.

    Args:
        obj: Neurogebra Expression or Model instance

    Returns:
        TensorFlow-compatible function or Keras Sequential model
    """
    check_tensorflow()
    if hasattr(obj, "layers"):
        return model_to_tensorflow(obj)

    expression = obj

    runtime_vars = [
        str(var) for var in expression.variables if str(var) not in expression.params
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
        str(var) for var in expression.variables if str(var) not in expression.params
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
