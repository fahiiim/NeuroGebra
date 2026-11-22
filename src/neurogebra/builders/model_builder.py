"""
ModelBuilder: Beginner-friendly neural network construction.

Provides intuitive interfaces for building ML models with educational
features built in.  v1.2.1 adds real forward/backward computation with
full Training Observatory integration.
"""

from __future__ import annotations

import copy
import time
from typing import Any

import numpy as np

# ======================================================================
# Activation helpers (vectorised NumPy implementations)
# ======================================================================


def _apply_activation(z: np.ndarray, name: str | None) -> np.ndarray:
    """Apply activation function element-wise."""
    if name is None or name == "linear":
        return z
    if name == "relu":
        return np.maximum(0, z)
    if name == "sigmoid":
        z_safe = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z_safe))
    if name == "tanh":
        return np.tanh(z)
    if name == "softmax":
        e = np.exp(z - np.max(z, axis=-1, keepdims=True))
        return e / np.sum(e, axis=-1, keepdims=True)
    if name == "leaky_relu":
        return np.where(z > 0, z, 0.01 * z)
    if name == "elu":
        return np.where(z > 0, z, np.exp(z) - 1)
    if name == "swish":
        sig = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
        return z * sig
    if name == "gelu":
        return 0.5 * z * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * z**3)))
    raise ValueError(f"Unsupported activation: '{name}'")


def _activation_derivative(
    z: np.ndarray, a: np.ndarray, name: str | None
) -> np.ndarray:
    """Compute derivative of activation w.r.t. pre-activation *z*."""
    if name is None or name == "linear":
        return np.ones_like(z)
    if name == "relu":
        return (z > 0).astype(z.dtype)
    if name == "sigmoid":
        return a * (1 - a)
    if name == "tanh":
        return 1 - a**2
    if name == "leaky_relu":
        return np.where(z > 0, 1.0, 0.01)
    if name == "elu":
        return np.where(z > 0, 1.0, a + 1.0)
    if name == "swish":
        sig = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
        return sig + z * sig * (1 - sig)
    if name == "gelu":
        u = np.sqrt(2.0 / np.pi) * (z + 0.044715 * z**3)
        tanh_u = np.tanh(u)
        du_dz = np.sqrt(2.0 / np.pi) * (1.0 + 3.0 * 0.044715 * z**2)
        return 0.5 * (1.0 + tanh_u) + 0.5 * z * (1.0 - tanh_u**2) * du_dz
    raise ValueError(f"Unsupported activation derivative: '{name}'")


class Layer:
    """
    Represents a single layer in a neural network.

    In v1.2.1 layers carry real weight/bias tensors and implement
    ``forward`` / ``backward`` for authentic computation, while
    retaining all educational metadata.

    Examples:
        >>> layer = Layer("dense", units=64, activation="relu")
        >>> layer.explain()
    """

    _DESCRIPTIONS = {
        "dense": (
            "Fully connected layer - every neuron connects to "
            "every neuron in next layer"
        ),
        "conv2d": "Convolutional layer - detects patterns in images",
        "dropout": (
            "Randomly turns off neurons during training to prevent overfitting"
        ),
        "flatten": "Converts multi-dimensional data to 1D for dense layers",
        "maxpool2d": ("Reduces image size by taking maximum values in regions"),
        "batchnorm": ("Normalizes activations in a layer to stabilize training"),
    }

    _USE_CASES = {
        "dense": ["Classification", "Regression", "Final output layer"],
        "conv2d": [
            "Image recognition",
            "Object detection",
            "Feature extraction",
        ],
        "dropout": ["Preventing overfitting", "Regularization"],
        "flatten": ["Bridging conv layers to dense layers"],
        "maxpool2d": ["Downsampling images", "Reducing computation"],
        "batchnorm": ["Faster training", "Reducing internal covariate shift"],
    }

    def __init__(
        self,
        layer_type: str,
        units: int | None = None,
        activation: str | None = None,
        **kwargs,
    ):
        """
        Create a layer.

        Args:
            layer_type: Type of layer ('dense', 'conv2d', 'dropout', etc.)
            units: Number of neurons/filters
            activation: Activation function name
            **kwargs: Additional layer-specific parameters
        """
        self.layer_type = layer_type
        self.units = units
        self.activation = activation
        self.params = kwargs

        # Educational metadata
        self.description = self._DESCRIPTIONS.get(self.layer_type, "Custom layer")
        self.best_for = self._USE_CASES.get(self.layer_type, ["General purpose"])

        # ---- Real computation state (initialised on first use) ----
        self.weights: np.ndarray | None = None  # (in_features, units)
        self.bias: np.ndarray | None = None  # (units,)
        self._initialised = False

        # Caches for backward pass
        self._last_input: np.ndarray | None = None
        self._last_z: np.ndarray | None = None  # pre-activation
        self._last_output: np.ndarray | None = None  # post-activation
        self._last_mask: np.ndarray | None = None  # dropout mask

        # Gradient accumulators
        self.grad_weights: np.ndarray | None = None
        self.grad_bias: np.ndarray | None = None
        self.gamma: np.ndarray | None = None
        self.beta: np.ndarray | None = None
        self.grad_gamma: np.ndarray | None = None
        self.grad_beta: np.ndarray | None = None
        self.running_mean: np.ndarray | None = None
        self.running_var: np.ndarray | None = None
        self.momentum: float = float(kwargs.get("momentum", 0.9))
        self.eps: float = float(kwargs.get("eps", 1e-5))

        # Adam state
        self._m_w: np.ndarray | None = None
        self._v_w: np.ndarray | None = None
        self._m_b: np.ndarray | None = None
        self._v_b: np.ndarray | None = None
        self._m_gamma: np.ndarray | None = None
        self._v_gamma: np.ndarray | None = None
        self._m_beta: np.ndarray | None = None
        self._v_beta: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Initialisation (He / Xavier)
    # ------------------------------------------------------------------

    def _init_params(self, input_dim: int, in_channels: int | None = None) -> None:
        if self.layer_type == "dense" and self.units is not None:
            scale = np.sqrt(2.0 / input_dim)  # He init
            self.weights = np.random.randn(input_dim, self.units) * scale
            self.bias = np.zeros(self.units)
            self._m_w = np.zeros_like(self.weights)
            self._v_w = np.zeros_like(self.weights)
            self._m_b = np.zeros_like(self.bias)
            self._v_b = np.zeros_like(self.bias)
        elif self.layer_type == "conv2d":
            filters = self.units or self.params.get("filters", 32)
            k = self.params.get("kernel_size", 3)
            c_in = in_channels if in_channels is not None else 1
            scale = np.sqrt(2.0 / (k * k * c_in))
            self.weights = np.random.randn(k, k, c_in, filters) * scale
            self.bias = np.zeros(filters)
            self._m_w = np.zeros_like(self.weights)
            self._v_w = np.zeros_like(self.weights)
            self._m_b = np.zeros_like(self.bias)
            self._v_b = np.zeros_like(self.bias)
        elif self.layer_type == "batchnorm":
            dim = self.units or input_dim
            self.gamma = np.ones(dim, dtype=np.float64)
            self.beta = np.zeros(dim, dtype=np.float64)
            self.running_mean = np.zeros(dim, dtype=np.float64)
            self.running_var = np.ones(dim, dtype=np.float64)
            self._m_gamma = np.zeros_like(self.gamma)
            self._v_gamma = np.zeros_like(self.gamma)
            self._m_beta = np.zeros_like(self.beta)
            self._v_beta = np.zeros_like(self.beta)
        self._initialised = True

    @property
    def n_params(self) -> int:
        total = 0
        if self.weights is not None:
            total += self.weights.size
        if self.bias is not None:
            total += self.bias.size
        if self.gamma is not None:
            total += self.gamma.size
        if self.beta is not None:
            total += self.beta.size
        return total

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Execute the forward pass and cache intermediates."""
        if self.layer_type == "dense":
            if not self._initialised:
                self._init_params(x.shape[-1])
            self._last_input = x
            self._last_z = x @ self.weights + self.bias
            self._last_output = _apply_activation(self._last_z, self.activation)
            return self._last_output

        if self.layer_type == "conv2d":
            if x.ndim == 3:
                x = x[..., np.newaxis]
            if not self._initialised:
                self._init_params(input_dim=x.shape[-1], in_channels=x.shape[-1])
            self._last_input = x
            k = self.params.get("kernel_size", 3)
            padding = str(self.params.get("padding", "valid")).lower()
            if padding == "same":
                p = k // 2
                x_pad = np.pad(x, ((0, 0), (p, p), (p, p), (0, 0)), mode="constant")
            else:
                x_pad = x
            self._last_padded = x_pad
            windows = np.lib.stride_tricks.sliding_window_view(
                x_pad, (k, k), axis=(1, 2)
            )
            z = (
                np.tensordot(windows, self.weights, axes=([3, 4, 5], [2, 0, 1]))
                + self.bias
            )
            self._last_z = z
            self._last_windows = windows
            self._last_output = _apply_activation(z, self.activation)
            return self._last_output

        if self.layer_type == "maxpool2d":
            if x.ndim == 3:
                x = x[..., np.newaxis]
            self._last_input = x
            P = self.params.get("pool_size", 2)
            N, H, W, C = x.shape
            H_trim = (H // P) * P
            W_trim = (W // P) * P
            x_crop = x[:, :H_trim, :W_trim, :]
            x_reshaped = (
                x_crop.reshape(N, H_trim // P, P, W_trim // P, P, C)
                .transpose(0, 1, 3, 5, 2, 4)
                .reshape(N, H_trim // P, W_trim // P, C, P * P)
            )
            out = x_reshaped.max(axis=-1)
            mask = x_reshaped == out[..., np.newaxis]
            self._last_maxpool_meta = (N, H, W, C, H_trim, W_trim, P, mask)
            self._last_output = out
            return out

        if self.layer_type == "dropout":
            rate = self.params.get("rate", 0.2)
            if training:
                self._last_mask = (np.random.rand(*x.shape) > rate).astype(x.dtype)
                return x * self._last_mask / (1 - rate)
            return x

        if self.layer_type == "flatten":
            self._last_input = x
            return x.reshape(x.shape[0], -1) if x.ndim > 2 else x

        if self.layer_type == "batchnorm":
            if not self._initialised:
                dim = self.units or x.shape[-1]
                self._init_params(input_dim=dim)
            self._last_input = x
            axes = tuple(range(x.ndim - 1))
            if training:
                mean = np.mean(x, axis=axes, keepdims=True)
                var = np.var(x, axis=axes, keepdims=True)
                x_norm = (x - mean) / np.sqrt(var + self.eps)

                batch_mean = np.mean(x, axis=axes)
                batch_var = np.var(x, axis=axes)
                self.running_mean = (
                    self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
                )
                self.running_var = (
                    self.momentum * self.running_var + (1 - self.momentum) * batch_var
                )

                self._last_x_norm = x_norm
                self._last_var = var
                self._last_mean = mean
                out = self.gamma * x_norm + self.beta
                self._last_output = out
                return out
            else:
                x_norm = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
                out = self.gamma * x_norm + self.beta
                self._last_output = out
                return out

        # Passthrough for unknown layer
        return x

    # ------------------------------------------------------------------
    # Backward pass
    # ------------------------------------------------------------------

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Compute gradients and return gradient for previous layer."""
        if self.layer_type == "dense":
            # For softmax activation, use the exact vector-Jacobian product:
            # dz = a * (grad_output - sum(grad_output * a, axis=-1, keepdims=True))
            if self.activation == "softmax":
                a = self._last_output
                dz = a * (grad_output - np.sum(grad_output * a, axis=-1, keepdims=True))
            else:
                da = _activation_derivative(
                    self._last_z, self._last_output, self.activation
                )
                dz = grad_output * da

            # Gradients: the upstream gradient is already normalized by loss reduction.
            # Do NOT divide by dz.shape[0] again (avoids double batch averaging).
            self.grad_weights = self._last_input.T @ dz
            self.grad_bias = np.sum(dz, axis=0)

            # Gradient for previous layer
            grad_input = dz @ self.weights.T
            return grad_input

        if self.layer_type == "conv2d":
            da = _activation_derivative(
                self._last_z, self._last_output, self.activation
            )
            dz = grad_output * da
            N, H_out, W_out, F = dz.shape
            k = self.params.get("kernel_size", 3)
            self.grad_bias = np.sum(dz, axis=(0, 1, 2))
            gw = np.tensordot(
                self._last_windows, dz, axes=([0, 1, 2], [0, 1, 2])
            ).transpose(1, 2, 0, 3)
            self.grad_weights = gw

            x_pad = self._last_padded
            grad_x_pad = np.zeros_like(x_pad)
            for di in range(k):
                for dj in range(k):
                    grad_x_pad[:, di : di + H_out, dj : dj + W_out, :] += (
                        dz @ self.weights[di, dj].T
                    )

            padding = str(self.params.get("padding", "valid")).lower()
            if padding == "same":
                p = k // 2
                grad_input = grad_x_pad[:, p:-p, p:-p, :] if p > 0 else grad_x_pad
            else:
                grad_input = grad_x_pad

            if self._last_input.ndim == 3 and grad_input.shape[-1] == 1:
                grad_input = grad_input.squeeze(-1)
            return grad_input

        if self.layer_type == "maxpool2d":
            N, H, W, C, H_trim, W_trim, P, mask = self._last_maxpool_meta
            denom = np.maximum(1, mask.sum(axis=-1, keepdims=True))
            g_reshaped = (grad_output[..., np.newaxis] * mask) / denom
            gx_crop = (
                g_reshaped.reshape(N, H_trim // P, W_trim // P, C, P, P)
                .transpose(0, 1, 4, 2, 5, 3)
                .reshape(N, H_trim, W_trim, C)
            )
            grad_input = np.zeros((N, H, W, C), dtype=gx_crop.dtype)
            grad_input[:, :H_trim, :W_trim, :] = gx_crop
            if self._last_input.ndim == 3 and grad_input.shape[-1] == 1:
                grad_input = grad_input.squeeze(-1)
            return grad_input

        if self.layer_type == "dropout":
            rate = self.params.get("rate", 0.2)
            if self._last_mask is not None:
                return grad_output * self._last_mask / (1 - rate)
            return grad_output

        if self.layer_type == "flatten":
            if self._last_input is not None:
                return grad_output.reshape(self._last_input.shape)
            return grad_output

        if self.layer_type == "batchnorm":
            axes = tuple(range(grad_output.ndim - 1))
            self.grad_beta = np.sum(grad_output, axis=axes)
            self.grad_gamma = np.sum(grad_output * self._last_x_norm, axis=axes)
            M = np.prod([grad_output.shape[i] for i in axes])
            inv_std = 1.0 / np.sqrt(self._last_var + self.eps)
            sum_dy = np.sum(grad_output, axis=axes, keepdims=True)
            sum_dy_xnorm = np.sum(
                grad_output * self._last_x_norm, axis=axes, keepdims=True
            )
            grad_input = (self.gamma * inv_std / M) * (
                M * grad_output - sum_dy - self._last_x_norm * sum_dy_xnorm
            )
            return grad_input

        return grad_output

    # ------------------------------------------------------------------
    # Weight update
    # ------------------------------------------------------------------

    def update_params(
        self,
        lr: float,
        optimizer: str = "adam",
        t: int = 1,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        """Apply gradient update to weights and bias."""
        if self.layer_type == "batchnorm":
            if self.gamma is not None and self.grad_gamma is not None:
                if optimizer == "sgd":
                    self.gamma -= lr * self.grad_gamma
                    self.beta -= lr * self.grad_beta
                elif optimizer == "adam":
                    self._m_gamma = beta1 * self._m_gamma + (1 - beta1) * self.grad_gamma
                    self._v_gamma = beta2 * self._v_gamma + (1 - beta2) * self.grad_gamma**2
                    m_hat_g = self._m_gamma / (1 - beta1**t)
                    v_hat_g = self._v_gamma / (1 - beta2**t)
                    self.gamma -= lr * m_hat_g / (np.sqrt(v_hat_g) + eps)

                    self._m_beta = beta1 * self._m_beta + (1 - beta1) * self.grad_beta
                    self._v_beta = beta2 * self._v_beta + (1 - beta2) * self.grad_beta**2
                    m_hat_b = self._m_beta / (1 - beta1**t)
                    v_hat_b = self._v_beta / (1 - beta2**t)
                    self.beta -= lr * m_hat_b / (np.sqrt(v_hat_b) + eps)
            return

        if self.weights is None or self.grad_weights is None:
            return

        if optimizer == "sgd":
            self.weights -= lr * self.grad_weights
            self.bias -= lr * self.grad_bias
        elif optimizer == "adam":
            self._m_w = beta1 * self._m_w + (1 - beta1) * self.grad_weights
            self._v_w = beta2 * self._v_w + (1 - beta2) * self.grad_weights**2
            m_hat_w = self._m_w / (1 - beta1**t)
            v_hat_w = self._v_w / (1 - beta2**t)
            self.weights -= lr * m_hat_w / (np.sqrt(v_hat_w) + eps)

            self._m_b = beta1 * self._m_b + (1 - beta1) * self.grad_bias
            self._v_b = beta2 * self._v_b + (1 - beta2) * self.grad_bias**2
            m_hat_b = self._m_b / (1 - beta1**t)
            v_hat_b = self._v_b / (1 - beta2**t)
            self.bias -= lr * m_hat_b / (np.sqrt(v_hat_b) + eps)

    def zero_grad(self) -> None:
        self.grad_weights = None
        self.grad_bias = None
        self.grad_gamma = None
        self.grad_beta = None
        self.grad_weights = None
        self.grad_bias = None

    # ------------------------------------------------------------------
    # Educational
    # ------------------------------------------------------------------

    def explain(self):
        """Explain what this layer does in plain language."""
        print(f"\n📚 {self.layer_type.upper()} Layer")
        print(f"   {self.description}")
        print("\n   Best used for:")
        for use_case in self.best_for:
            print(f"   • {use_case}")

        if self.units:
            print(f"\n   Neurons/Units: {self.units}")
        if self.activation:
            print(f"   Activation: {self.activation}")
        if self.n_params > 0:
            print(f"   Parameters: {self.n_params:,}")
        for key, val in self.params.items():
            print(f"   {key}: {val}")

    def get_weight_stats(self) -> dict[str, Any] | None:
        """Return summary statistics for this layer's weights."""
        if self.weights is None:
            return None
        w = self.weights.ravel()
        return {
            "mean": float(np.mean(w)),
            "std": float(np.std(w)),
            "min": float(np.min(w)),
            "max": float(np.max(w)),
            "norm_l2": float(np.linalg.norm(w)),
            "zeros_pct": float(np.sum(np.abs(w) < 1e-6) / w.size * 100),
            "size": int(w.size),
        }

    def get_gradient_stats(self) -> dict[str, Any] | None:
        """Return summary statistics for this layer's weight gradients."""
        if self.grad_weights is None:
            return None
        g = self.grad_weights.ravel()
        return {
            "mean": float(np.mean(g)),
            "std": float(np.std(g)),
            "min": float(np.min(g)),
            "max": float(np.max(g)),
            "norm_l2": float(np.linalg.norm(g)),
        }

    def __repr__(self):
        parts = [f"{self.layer_type}"]
        if self.units:
            parts.append(f"units={self.units}")
        if self.activation:
            parts.append(f"activation={self.activation}")
        for key, val in self.params.items():
            parts.append(f"{key}={val}")
        return f"Layer({', '.join(parts)})"


class ModelBuilder:
    """
    Build neural networks with an educational, intuitive interface.

    ModelBuilder makes it easy for beginners to:
    - Understand what they're building
    - Get guidance on architecture choices
    - See what each layer does
    - Learn best practices

    Examples:
        >>> from neurogebra.builders import ModelBuilder
        >>> builder = ModelBuilder()
        >>> model = builder.Sequential([
        ...     builder.Dense(128, activation="relu"),
        ...     builder.Dropout(0.2),
        ...     builder.Dense(10, activation="softmax")
        ... ])
        >>> model.summary()
    """

    def __init__(self, craft: Any | None = None):
        """
        Initialize ModelBuilder.

        Args:
            craft: NeuroCraft instance for expression access.
                   If None, a default one is created internally.
        """
        self.craft = craft
        self._templates = self._load_templates()

    def _get_craft(self):
        """Lazily initialize craft if needed."""
        if self.craft is None:
            from neurogebra.core.neurocraft import NeuroCraft

            self.craft = NeuroCraft(educational_mode=False)
        return self.craft

    def _load_templates(self) -> dict[str, dict[str, Any]]:
        """Load pre-built model templates for common tasks."""
        return {
            "simple_classifier": {
                "description": "Basic neural network for classification",
                "layers": [
                    {"type": "dense", "units": 128, "activation": "relu"},
                    {"type": "dropout", "rate": 0.2},
                    {"type": "dense", "units": 64, "activation": "relu"},
                    {"type": "dense", "units": 10, "activation": "softmax"},
                ],
                "good_for": [
                    "MNIST",
                    "CIFAR-10",
                    "Simple tabular data",
                ],
            },
            "image_classifier": {
                "description": "CNN for image classification",
                "layers": [
                    {
                        "type": "conv2d",
                        "filters": 32,
                        "kernel_size": 3,
                        "activation": "relu",
                    },
                    {"type": "maxpool2d", "pool_size": 2},
                    {
                        "type": "conv2d",
                        "filters": 64,
                        "kernel_size": 3,
                        "activation": "relu",
                    },
                    {"type": "maxpool2d", "pool_size": 2},
                    {"type": "flatten"},
                    {"type": "dense", "units": 128, "activation": "relu"},
                    {"type": "dense", "units": 10, "activation": "softmax"},
                ],
                "good_for": [
                    "Image recognition",
                    "Object classification",
                ],
            },
            "regression": {
                "description": "Network for predicting continuous values",
                "layers": [
                    {"type": "dense", "units": 64, "activation": "relu"},
                    {"type": "dense", "units": 32, "activation": "relu"},
                    {"type": "dense", "units": 1, "activation": "linear"},
                ],
                "good_for": [
                    "House price prediction",
                    "Stock prices",
                    "Any regression task",
                ],
            },
            "binary_classifier": {
                "description": "Network for binary yes/no classification",
                "layers": [
                    {"type": "dense", "units": 64, "activation": "relu"},
                    {"type": "dropout", "rate": 0.3},
                    {"type": "dense", "units": 32, "activation": "relu"},
                    {"type": "dense", "units": 1, "activation": "sigmoid"},
                ],
                "good_for": [
                    "Spam detection",
                    "Medical diagnosis",
                    "Sentiment analysis",
                ],
            },
        }

    # ---- Layer creation methods ----

    def Dense(
        self,
        units: int,
        activation: str | None = None,
        input_shape: tuple | None = None,
        **kwargs,
    ) -> Layer:
        """
        Create a fully connected (dense) layer.

        Args:
            units: Number of neurons
            activation: Activation function name
            input_shape: Shape of input (only for first layer)

        Returns:
            Layer instance

        Examples:
            >>> builder = ModelBuilder()
            >>> layer = builder.Dense(128, activation="relu")
            >>> layer.explain()  # Learn what it does
        """
        extra = {**kwargs}
        if input_shape is not None:
            extra["input_shape"] = input_shape
        return Layer("dense", units=units, activation=activation, **extra)

    def Conv2D(
        self,
        filters: int,
        kernel_size: int = 3,
        activation: str | None = None,
        **kwargs,
    ) -> Layer:
        """
        Create a 2D convolutional layer for images.

        Args:
            filters: Number of filters / feature detectors
            kernel_size: Size of the filter window
            activation: Activation function

        Returns:
            Layer instance
        """
        return Layer(
            "conv2d",
            units=filters,
            activation=activation,
            kernel_size=kernel_size,
            **kwargs,
        )

    def Dropout(self, rate: float = 0.2) -> Layer:
        """
        Create a dropout layer for regularization.

        Args:
            rate: Fraction of neurons to drop (0.0 to 1.0)

        Returns:
            Layer instance
        """
        return Layer("dropout", rate=rate)

    def BatchNorm(self) -> Layer:
        """Create a batch normalization layer."""
        return Layer("batchnorm")

    def MaxPooling2D(self, pool_size: int = 2) -> Layer:
        """Create a max pooling layer."""
        return Layer("maxpool2d", pool_size=pool_size)

    def Flatten(self) -> Layer:
        """Create a flatten layer."""
        return Layer("flatten")

    # ---- Model construction ----

    def Sequential(self, layers: list[Layer], name: str | None = None) -> Model:
        """
        Build a sequential model (layers stacked one after another).

        Args:
            layers: List of Layer instances
            name: Optional name for the model

        Returns:
            Model instance ready for compilation and training

        Examples:
            >>> builder = ModelBuilder()
            >>> model = builder.Sequential([
            ...     builder.Dense(128, activation="relu"),
            ...     builder.Dropout(0.2),
            ...     builder.Dense(10, activation="softmax")
            ... ])
            >>> model.summary()
        """
        return Model(layers=layers, name=name, craft=self._get_craft())

    def from_template(
        self,
        template_name: str,
        customize: dict | None = None,
    ) -> Model:
        """
        Create a model from a pre-built template.

        Args:
            template_name: Name of template (e.g. 'simple_classifier')
            customize: Optional customizations (not yet implemented)

        Returns:
            Model instance

        Examples:
            >>> builder = ModelBuilder()
            >>> model = builder.from_template("simple_classifier")
            >>> model.explain_architecture()
        """
        if template_name not in self._templates:
            available = ", ".join(self._templates.keys())
            raise ValueError(
                f"Template '{template_name}' not found.\n"
                f"Available templates: {available}\n"
                f"Use builder.list_templates() to see details."
            )

        template = self._templates[template_name]

        # Deep-copy so repeated calls don't mutate the template
        layer_specs = copy.deepcopy(template["layers"])

        layers = []
        for layer_spec in layer_specs:
            layer_type = layer_spec.pop("type")

            if layer_type == "dense":
                layers.append(self.Dense(**layer_spec))
            elif layer_type == "conv2d":
                layers.append(self.Conv2D(**layer_spec))
            elif layer_type == "dropout":
                layers.append(self.Dropout(**layer_spec))
            elif layer_type == "maxpool2d":
                layers.append(self.MaxPooling2D(**layer_spec))
            elif layer_type == "flatten":
                layers.append(self.Flatten())
            elif layer_type == "batchnorm":
                layers.append(self.BatchNorm())

        model = self.Sequential(layers, name=template_name)
        model.template_info = template

        return model

    def list_templates(self):
        """Show all available model templates with descriptions."""
        print("\n🏗️  Available Model Templates:\n")

        for name, template in self._templates.items():
            print(f"  📦 {name}")
            print(f"     {template['description']}")
            print(f"     Good for: {', '.join(template['good_for'])}")
            print()

    def suggest_architecture(
        self,
        task: str,
        input_shape: tuple,
        output_size: int,
    ):
        """
        Get architecture suggestions based on your task.

        Args:
            task: 'classification', 'regression', 'image_classification', etc.
            input_shape: Shape of your input data
            output_size: Number of outputs

        Examples:
            >>> builder = ModelBuilder()
            >>> builder.suggest_architecture(
            ...     task="image_classification",
            ...     input_shape=(28, 28, 1),
            ...     output_size=10
            ... )
        """
        print(f"\n💡 Architecture Suggestions for {task}:\n")

        if task in ("classification", "image_classification"):
            if len(input_shape) == 3:
                print("   Recommended: Convolutional Neural Network (CNN)")
                print("   Why: CNNs are designed to detect patterns in images")
                print("\n   Suggested architecture:")
                print("   1. Conv2D(32, kernel_size=3) + ReLU")
                print("   2. MaxPooling2D()")
                print("   3. Conv2D(64, kernel_size=3) + ReLU")
                print("   4. MaxPooling2D()")
                print("   5. Flatten()")
                print("   6. Dense(128) + ReLU")
                print("   7. Dropout(0.3)")
                print(f"   8. Dense({output_size}) + Softmax")
                print("\n   Use: builder.from_template('image_classifier')")
            else:
                print("   Recommended: Simple Feedforward Network")
                print("   Why: Efficient for structured/tabular data")
                print("\n   Suggested architecture:")
                print("   1. Dense(128) + ReLU")
                print("   2. Dropout(0.2)")
                print("   3. Dense(64) + ReLU")
                print(f"   4. Dense({output_size}) + Softmax")
                print("\n   Use: builder.from_template('simple_classifier')")

        elif task == "regression":
            print("   Recommended: Regression Network")
            print("   Why: Predicts continuous values")
            print("\n   Suggested architecture:")
            print("   1. Dense(64) + ReLU")
            print("   2. Dense(32) + ReLU")
            print("   3. Dense(1) + Linear (no activation)")
            print("\n   Use: builder.from_template('regression')")

        elif task == "binary_classification":
            print("   Recommended: Binary Classifier")
            print("   Why: Two-class output with sigmoid")
            print("\n   Suggested architecture:")
            print("   1. Dense(64) + ReLU")
            print("   2. Dropout(0.3)")
            print("   3. Dense(32) + ReLU")
            print("   4. Dense(1) + Sigmoid")
            print("\n   Use: builder.from_template('binary_classifier')")

        print("\n   💡 Tip: Start with these suggestions, then experiment!")


class Model:
    """
    Represents a complete neural network model.

    Educational model class that provides beginner-friendly interfaces
    for understanding, compiling, training, and evaluating models.

    In v1.2.1 the model performs **real forward/backward computation**
    through its layers and integrates with the Training Observatory
    for colourful, in-depth mathematical logging.
    """

    def __init__(
        self,
        layers: list[Layer],
        name: str | None = None,
        craft=None,
    ):
        self.layers = layers
        self.name = name or "UntitledModel"
        self.craft = craft
        self.template_info: dict | None = None

        # Training state
        self.is_compiled = False
        self.loss_function = None
        self.loss_name: str | None = None
        self.optimizer: str | None = None
        self.learning_rate: float = 0.001
        self.metrics: list[str] = []

        # Training Observatory
        self._logger = None
        self._log_config = None

        # History
        self.history: dict[str, list[float]] = {
            "loss": [],
            "accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }

    def _get_craft(self):
        if self.craft is None:
            from neurogebra.core.neurocraft import NeuroCraft

            self.craft = NeuroCraft(educational_mode=False)
        return self.craft

    # ------------------------------------------------------------------
    # Real forward / backward
    # ------------------------------------------------------------------

    def _real_forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """Chain forward passes through all layers."""
        out = X
        for i, layer in enumerate(self.layers):
            t0 = time.time()
            out = layer.forward(out, training=training)
            elapsed = time.time() - t0

            if self._logger:
                self._logger.on_layer_forward(
                    layer_index=i,
                    layer_name=f"{layer.layer_type}_{i}",
                    input_data=layer._last_input,
                    output_data=layer._last_output,
                    weights=layer.weights,
                    bias=layer.bias,
                    formula=self._layer_formula(layer, i),
                    elapsed=elapsed,
                )
        return out

    def _real_backward(self, grad: np.ndarray) -> None:
        """Backpropagate gradient through layers in reverse."""
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            t0 = time.time()
            grad = layer.backward(grad)
            elapsed = time.time() - t0

            if self._logger:
                self._logger.on_layer_backward(
                    layer_index=i,
                    layer_name=f"{layer.layer_type}_{i}",
                    grad_output=grad,
                    grad_weights=layer.grad_weights,
                    grad_bias=layer.grad_bias,
                    formula=self._layer_grad_formula(layer, i),
                    elapsed=elapsed,
                )

    def _update_params(self, t: int) -> None:
        """Apply weight updates on all layers."""
        for i, layer in enumerate(self.layers):
            old_w = layer.weights.copy() if layer.weights is not None else None
            layer.update_params(
                lr=self.learning_rate,
                optimizer=self.optimizer or "adam",
                t=t,
            )
            if self._logger and old_w is not None and layer.weights is not None:
                delta = float(np.linalg.norm(layer.weights - old_w))
                self._logger.on_weight_updated(
                    param_name=f"W_{layer.layer_type}_{i}",
                    old_value=float(np.mean(old_w)),
                    new_value=float(np.mean(layer.weights)),
                    delta_norm=delta,
                )

    @staticmethod
    def _layer_formula(layer: Layer, idx: int) -> str:
        if layer.layer_type == "dense":
            act = layer.activation or "linear"
            return f"a{idx} = {act}(W{idx}·x + b{idx})"
        if layer.layer_type == "dropout":
            return f"dropout(p={layer.params.get('rate', 0.2)})"
        return layer.layer_type

    @staticmethod
    def _layer_grad_formula(layer: Layer, idx: int) -> str:
        if layer.layer_type == "dense":
            act = layer.activation or "linear"
            return f"∂L/∂W{idx} = ∂L/∂a{idx} ⊙ {act}'(z{idx}) · a{idx - 1}ᵀ"
        return ""

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_loss(
        predictions: np.ndarray, targets: np.ndarray, loss_name: str
    ) -> tuple:
        """Return (loss_scalar, grad_wrt_predictions)."""
        eps = 1e-12
        n = predictions.shape[0]

        if loss_name in ("cross_entropy", "categorical_crossentropy"):
            # Handle both sparse labels and one-hot targets
            if targets.ndim == 1 or (targets.ndim == 2 and targets.shape[1] == 1):
                targets_idx = targets.ravel().astype(int)
                num_classes = predictions.shape[-1]
                if np.any(targets_idx < 0) or np.any(targets_idx >= num_classes):
                    raise ValueError(
                        f"Target class index out of bounds [0, {num_classes - 1}]"
                    )
                one_hot = np.zeros_like(predictions)
                one_hot[np.arange(n), targets_idx] = 1.0
            elif targets.shape == predictions.shape:
                one_hot = targets.astype(predictions.dtype)
            else:
                raise ValueError(
                    f"Target shape {targets.shape} incompatible with predictions shape {predictions.shape} for {loss_name}"
                )

            p = np.clip(predictions, eps, 1.0)
            loss = -float(np.mean(np.sum(one_hot * np.log(p), axis=-1)))
            # True gradient w.r.t. predictions: -one_hot / (p * n)
            # When passed into softmax backward, a * (g - sum(g*a)) simplifies analytically to (p - one_hot) / n.
            grad = -one_hot / (p * n)
            return loss, grad

        # For regression & binary cross-entropy, align targets shape
        if targets.shape != predictions.shape:
            if targets.size == predictions.size:
                t = targets.reshape(predictions.shape)
            elif (
                predictions.ndim == 2
                and predictions.shape[-1] > 1
                and (targets.ndim == 1 or (targets.ndim == 2 and targets.shape[1] == 1))
            ):
                # Integer class labels passed with multi-class output (e.g. MSE on class labels)
                targets_idx = targets.ravel().astype(int)
                num_classes = predictions.shape[-1]
                t = np.zeros_like(predictions)
                if np.all((targets_idx >= 0) & (targets_idx < num_classes)):
                    t[np.arange(n), targets_idx] = 1.0
                else:
                    raise ValueError(
                        f"Target class index out of bounds [0, {num_classes - 1}]"
                    )
            else:
                raise ValueError(
                    f"Target shape {targets.shape} incompatible with predictions shape {predictions.shape} for {loss_name}"
                )
        else:
            t = targets

        if loss_name in ("mse", "mean_squared_error"):
            diff = predictions - t
            loss = float(np.mean(diff**2))
            grad = 2.0 * diff / predictions.size
            return loss, grad

        if loss_name in ("mae", "mean_absolute_error"):
            diff = predictions - t
            loss = float(np.mean(np.abs(diff)))
            grad = np.sign(diff) / predictions.size
            return loss, grad

        if loss_name == "binary_crossentropy":
            p = np.clip(predictions, eps, 1 - eps)
            loss = -float(np.mean(t * np.log(p) + (1 - t) * np.log(1 - p)))
            grad = (-t / p + (1 - t) / (1 - p)) / predictions.size
            return loss, grad

        raise ValueError(f"Unsupported loss function: '{loss_name}'")

    @staticmethod
    def _compute_accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
        if targets.ndim > 1 and targets.shape[-1] > 1:
            true_labels = np.argmax(targets, axis=-1)
        else:
            true_labels = targets.ravel().astype(int)

        if predictions.ndim > 1 and predictions.shape[-1] > 1:
            pred_labels = np.argmax(predictions, axis=-1)
        else:
            pred_labels = (predictions.ravel() > 0.5).astype(int)

        if len(pred_labels) == 0 or len(true_labels) == 0:
            return 0.0
        n = min(len(pred_labels), len(true_labels))
        return float(np.mean(pred_labels[:n] == true_labels[:n]))

    # ------------------------------------------------------------------
    # Model info dict (for logger)
    # ------------------------------------------------------------------

    def _model_info(self) -> dict[str, Any]:
        total_params = sum(layer.n_params for layer in self.layers)
        return {
            "name": self.name,
            "num_layers": len(self.layers),
            "total_params": total_params,
            "loss": self.loss_name,
            "optimizer": self.optimizer,
            "lr": self.learning_rate,
            "layers": [repr(layer) for layer in self.layers],
        }

    # ------------------------------------------------------------------
    # summary / explain
    # ------------------------------------------------------------------

    def summary(self, educational: bool = True):
        print(f"\n{'=' * 60}")
        print(f"  Model: {self.name}")
        print(f"{'=' * 60}\n")

        total_params = 0
        prev_units: int | None = None

        for i, layer in enumerate(self.layers, 1):
            print(f"  Layer {i}: {layer}")

            if educational:
                print(f"           {layer.description}")

            if layer.layer_type == "dense" and layer.units:
                input_size = prev_units if prev_units else 100
                params = input_size * layer.units + layer.units
                total_params += params
                print(f"           Parameters: ~{params:,}")

            if layer.units is not None:
                prev_units = layer.units
            print()

        print(f"{'=' * 60}")
        print(f"  Total Parameters: ~{total_params:,}")
        print(f"{'=' * 60}\n")

        if educational and self.template_info:
            print("  📚 About this architecture:")
            print(f"     {self.template_info['description']}")
            print(f"     Good for: {', '.join(self.template_info['good_for'])}")
            print()

    def explain_architecture(self):
        print(f"\n🎓 Understanding Your Model: {self.name}\n")
        for i, layer in enumerate(self.layers, 1):
            print(f"{'=' * 60}")
            print(f"Layer {i}: {layer.layer_type.upper()}")
            print(f"{'=' * 60}")
            layer.explain()
            if i < len(self.layers):
                print("\n   ⬇️  Passes data to next layer")
            print()

    # ------------------------------------------------------------------
    # compile / fit / predict / evaluate
    # ------------------------------------------------------------------

    def compile(
        self,
        loss: str = "mse",
        optimizer: str = "adam",
        metrics: list[str] | None = None,
        learning_rate: float = 0.001,
        log_level: str | None = None,
        log_config=None,
    ):
        """
        Configure the model for training.

        Args:
            loss: Loss function name.
            optimizer: Optimizer name ('adam', 'sgd').
            metrics: List of metrics to track.
            learning_rate: Learning rate.
            log_level: Observatory log level ('silent','basic','detailed','expert','debug').
            log_config: A ``LogConfig`` instance for full control.
        """
        print("\n⚙️  Compiling model...")

        self.loss_name = loss
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.metrics = metrics or ["accuracy"]

        try:
            craft = self._get_craft()
            self.loss_function = craft.get(loss)
        except (KeyError, Exception):
            self.loss_function = None

        # --- Training Observatory setup ---
        if log_level or log_config:
            from neurogebra.logging.config import LogConfig
            from neurogebra.logging.logger import LogLevel, TrainingLogger
            from neurogebra.logging.terminal_display import TerminalDisplay

            if log_config is None:
                lvl = getattr(LogLevel, (log_level or "BASIC").upper(), LogLevel.BASIC)
                log_config = LogConfig(level=lvl)
                if lvl >= LogLevel.EXPERT:
                    log_config.show_formulas = True
                    log_config.show_gradients = True
                    log_config.show_weights = True
                    log_config.show_activations = True
                    log_config.show_timing = True
                    log_config.show_images = True

            self._log_config = log_config
            self._logger = TrainingLogger(level=log_config.level)
            self._logger.add_backend(TerminalDisplay(log_config))

            # Optional exporters
            if "json" in log_config.export_formats:
                from neurogebra.logging.exporters import JSONExporter

                self._logger.add_backend(
                    JSONExporter(f"{log_config.export_dir}/training_log.json")
                )
            if "csv" in log_config.export_formats:
                from neurogebra.logging.exporters import CSVExporter

                self._logger.add_backend(
                    CSVExporter(f"{log_config.export_dir}/metrics.csv")
                )
            if "html" in log_config.export_formats:
                from neurogebra.logging.exporters import HTMLExporter

                self._logger.add_backend(
                    HTMLExporter(f"{log_config.export_dir}/report.html")
                )
            if "markdown" in log_config.export_formats:
                from neurogebra.logging.exporters import MarkdownExporter

                self._logger.add_backend(
                    MarkdownExporter(f"{log_config.export_dir}/report.md")
                )

        print(f"   Loss: {loss}")
        print(f"   Optimizer: {optimizer} (lr={learning_rate})")
        print(f"   Metrics: {', '.join(self.metrics)}")
        if self._logger:
            print(f"   Observatory: level={self._log_config.level.name}")

        self.is_compiled = True
        print("   ✅ Model compiled successfully!\n")

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 10,
        batch_size: int = 32,
        validation_split: float = 0.2,
        verbose: bool = True,
        visualize: bool = False,
    ) -> dict[str, list[float]]:
        """
        Train the model on data.

        Uses **real forward/backward** computation with full
        Training Observatory integration when a log_level is set.
        Falls back to EducationalTrainer when no Observatory is active.
        """
        if not self.is_compiled:
            raise RuntimeError(
                "Model must be compiled before training!\nCall model.compile() first."
            )

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        # If Observatory is active, use real training loop
        if self._logger:
            return self._real_fit(X, y, epochs, batch_size, validation_split)

        # Otherwise use educational trainer (backward compat)
        print(f"\n🚀 Training {self.name}...\n")
        print(f"   Training samples: {len(X)}")
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {batch_size}\n")

        from neurogebra.training.educational_trainer import EducationalTrainer

        trainer = EducationalTrainer(
            model=self,
            verbose=verbose,
            visualize=visualize,
        )
        history = trainer.train(
            X=X,
            y=y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
        )
        self.history = history
        print("\n✅ Training complete!")
        print(f"   Final loss: {history['loss'][-1]:.4f}")
        if history.get("accuracy"):
            print(f"   Final accuracy: {history['accuracy'][-1]:.4f}")
        return history

    # ------------------------------------------------------------------
    # Real training loop with Observatory
    # ------------------------------------------------------------------

    def _real_fit(self, X, y, epochs, batch_size, validation_split):
        from neurogebra.logging.formula_renderer import FormulaRenderer
        from neurogebra.logging.health_checks import SmartHealthChecker
        from neurogebra.logging.image_logger import ImageLogger
        from neurogebra.logging.monitors import (
            ActivationMonitor,
            GradientMonitor,
            PerformanceMonitor,
            WeightMonitor,
        )

        grad_mon = GradientMonitor()
        weight_mon = WeightMonitor()
        act_mon = ActivationMonitor()
        perf_mon = PerformanceMonitor()
        checker = SmartHealthChecker()
        renderer = FormulaRenderer()
        img_logger = ImageLogger()

        # Split
        if validation_split > 0 and len(X) > 1:
            split_idx = int(len(X) * (1 - validation_split))
            split_idx = max(1, min(split_idx, len(X) - 1))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
        else:
            X_train = X
            X_val = np.empty((0, *X.shape[1:]), dtype=X.dtype)
            y_train = y
            y_val = np.empty((0, *y.shape[1:]), dtype=y.dtype)

        self._logger.on_train_start(
            model_info=self._model_info(),
            total_epochs=epochs,
            total_samples=len(X_train),
            batch_size=batch_size,
        )

        # Render full model equation once
        if self._log_config and self._log_config.show_formulas:
            specs = [
                {
                    "type": layer.layer_type,
                    "activation": layer.activation,
                    "rate": layer.params.get("rate"),
                }
                for layer in self.layers
            ]
            eq = renderer.full_model_equation(specs)
            renderer.render(eq, title="Full Model Equation", style="cyan")
            renderer.render_loss(self.loss_name or "mse")

        # Check for image data
        if (
            self._log_config
            and self._log_config.show_images
            and img_logger.is_image_data(X_train)
        ):
            img_logger.render_image(X_train[0], title="Sample Input Image")

        global_step = 0

        for epoch in range(epochs):
            self._logger.on_epoch_start(epoch)
            epoch_t0 = time.time()

            # Shuffle
            idx = np.random.permutation(len(X_train))
            Xs, ys = X_train[idx], y_train[idx]

            epoch_losses = []
            epoch_accs = []

            batch_starts = list(range(0, len(X_train), batch_size))
            for b, s in enumerate(batch_starts):
                e = min(s + batch_size, len(X_train))
                xb, yb = Xs[s:e], ys[s:e]

                global_step += 1
                self._logger.on_batch_start(b, epoch=epoch, X_batch=xb, y_batch=yb)

                # Forward
                preds = self._real_forward(xb, training=True)
                loss, grad = self._compute_loss(preds, yb, self.loss_name or "mse")
                acc = self._compute_accuracy(preds, yb)

                epoch_losses.append(loss)
                epoch_accs.append(acc)

                # Backward
                self._real_backward(grad)

                # Monitor gradients & activations
                grad_norms = {}
                for i, layer in enumerate(self.layers):
                    lname = f"{layer.layer_type}_{i}"
                    if layer.grad_weights is not None:
                        gs = grad_mon.record(lname, layer.grad_weights, layer.weights)
                        grad_norms[lname] = gs["norm_l2"]
                    if layer.weights is not None:
                        weight_mon.record(lname, layer.weights)
                    if layer._last_output is not None:
                        act_mon.record(
                            lname, layer._last_output, layer.activation or "linear"
                        )

                # Update weights
                self._update_params(t=global_step)

                # Zero grads
                for layer in self.layers:
                    layer.zero_grad()

                self._logger.on_batch_end(
                    b, epoch=epoch, loss=loss, metrics={"accuracy": acc}
                )

            # Validation
            if len(X_val) > 0:
                val_preds = self._real_forward(X_val, training=False)
                val_loss, _ = self._compute_loss(val_preds, y_val, self.loss_name or "mse")
                val_acc = self._compute_accuracy(val_preds, y_val)
            else:
                val_loss = 0.0
                val_acc = 0.0

            epoch_time = time.time() - epoch_t0
            perf_mon.record_epoch_time(epoch_time)

            avg_loss = float(np.mean(epoch_losses))
            avg_acc = float(np.mean(epoch_accs))

            self.history["loss"].append(avg_loss)
            self.history["accuracy"].append(avg_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_accuracy"].append(val_acc)

            self._logger.on_epoch_end(
                epoch,
                metrics={
                    "loss": avg_loss,
                    "accuracy": avg_acc,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                },
            )

            # Health checks
            if self._log_config and self._log_config.show_health_checks:
                if (epoch + 1) % self._log_config.health_check_interval == 0:
                    weight_stats_dict = {}
                    for i, layer in enumerate(self.layers):
                        lname = f"{layer.layer_type}_{i}"
                        ws = layer.get_weight_stats()
                        if ws:
                            weight_stats_dict[lname] = ws
                    alerts = checker.run_all(
                        epoch=epoch,
                        train_losses=self.history["loss"],
                        val_losses=self.history["val_loss"],
                        train_accs=self.history["accuracy"],
                        gradient_norms=grad_norms,
                        weight_stats=weight_stats_dict,
                    )
                    for alert in alerts:
                        self._logger.on_health_check(
                            check_name=alert.check_name,
                            severity=alert.severity,
                            message=alert.message,
                            recommendations=alert.recommendations,
                        )

        # Final
        final_metrics = {
            "loss": self.history["loss"][-1] if self.history["loss"] else 0,
            "accuracy": self.history["accuracy"][-1] if self.history["accuracy"] else 0,
            "val_loss": self.history["val_loss"][-1] if self.history["val_loss"] else 0,
            "val_accuracy": self.history["val_accuracy"][-1]
            if self.history["val_accuracy"]
            else 0,
        }
        self._logger.on_train_end(final_metrics=final_metrics)

        # Save exports
        for backend in self._logger._backends:
            if hasattr(backend, "save"):
                backend.save()

        return self.history

    # ------------------------------------------------------------------
    # predict / evaluate
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions using the model's forward pass."""
        X = np.asarray(X, dtype=np.float64)
        return self._real_forward(X, training=False)

    def train_on_batch(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Execute a single forward, backward, and optimization step on a single batch.

        Args:
            X: Input batch features
            y: Target batch labels/values

        Returns:
            Scalar loss for the batch
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if not hasattr(self, "_step_counter"):
            self._step_counter = 1

        preds = self._real_forward(X, training=True)
        loss, grad = self._compute_loss(preds, y, self.loss_name or "mse")
        self._real_backward(grad)
        self._update_params(t=self._step_counter)
        self._step_counter += 1

        for layer in self.layers:
            layer.zero_grad()

        return float(loss)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        predictions = self.predict(X)
        loss, _ = self._compute_loss(predictions, y, self.loss_name or "mse")
        acc = self._compute_accuracy(predictions, y)
        print(f"\n📊 Evaluation — loss: {loss:.4f}, accuracy: {acc:.4f}")
        return {"loss": loss, "accuracy": acc}

    def save(self, filepath: str):
        """Save model architecture and trained weights to file."""
        import pickle

        with open(filepath, "wb") as f:
            pickle.dump(
                {
                    "name": self.name,
                    "layers": [
                        {
                            "type": layer.layer_type,
                            "units": layer.units,
                            "activation": layer.activation,
                            "params": layer.params,
                            "weights": layer.weights,
                            "bias": layer.bias,
                            "gamma": getattr(layer, "gamma", None),
                            "beta": getattr(layer, "beta", None),
                            "running_mean": getattr(layer, "running_mean", None),
                            "running_var": getattr(layer, "running_var", None),
                        }
                        for layer in self.layers
                    ],
                    "history": self.history,
                    "loss_name": self.loss_name,
                    "optimizer": self.optimizer,
                    "learning_rate": self.learning_rate,
                },
                f,
            )
        print(f"💾 Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> Model:
        """Load a saved model from file."""
        import pickle

        with open(filepath, "rb") as f:
            data = pickle.load(f)

        layers = []
        for l_info in data["layers"]:
            layer = Layer(
                layer_type=l_info["type"],
                units=l_info["units"],
                activation=l_info["activation"],
                **l_info.get("params", {}),
            )
            layer.weights = l_info.get("weights")
            layer.bias = l_info.get("bias")
            if l_info.get("gamma") is not None:
                layer.gamma = l_info["gamma"]
            if l_info.get("beta") is not None:
                layer.beta = l_info["beta"]
            if l_info.get("running_mean") is not None:
                layer.running_mean = l_info["running_mean"]
            if l_info.get("running_var") is not None:
                layer.running_var = l_info["running_var"]
            layer._initialised = True
            layers.append(layer)

        model = cls(layers=layers, name=data.get("name", "loaded_model"))
        model.loss_name = data.get("loss_name")
        model.optimizer = data.get("optimizer", "adam")
        model.learning_rate = data.get("learning_rate", 0.001)
        model.history = data.get(
            "history", {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
        )
        model.is_compiled = True
        return model

    def plot_history(self):
        from neurogebra.viz.plotting import plot_training_history

        plot_training_history(self.history)

    def to_pytorch(self):
        from neurogebra.bridges.pytorch_bridge import to_pytorch

        return to_pytorch(self)

    def to_tensorflow(self):
        from neurogebra.bridges.tensorflow_bridge import to_tensorflow

        return to_tensorflow(self)

    def __repr__(self):
        return (
            f"Model(name='{self.name}', layers={len(self.layers)}, "
            f"compiled={self.is_compiled})"
        )
