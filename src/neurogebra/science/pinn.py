"""
Physics-Informed Neural Networks (PINN) for NeuroGebra.

Implements deep learning solvers for Ordinary and Partial Differential Equations (ODEs/PDEs)
enforcing physical conservation laws through residual minimization.
"""

from __future__ import annotations

import inspect
from typing import Callable

import numpy as np


class PINN:
    """
    Physics-Informed Neural Network (PINN).

    Solves differential equations by training a neural network that simultaneously
    fits boundary/initial conditions and minimizes differential equation residuals
    at sampled collocation points in the domain.

    Args:
        layer_sizes: List of integers representing layer widths (e.g., [2, 32, 32, 1] for 2D domain to 1D field).
        activation: Activation function name ('tanh', 'sin', 'gelu', 'relu').
        seed: Random seed.
    """

    def __init__(
        self,
        layer_sizes: list[int] | tuple[int, ...],
        activation: str = "tanh",
        seed: int = 42,
    ):
        self.layer_sizes = list(layer_sizes)
        if len(self.layer_sizes) < 2 or any(
            not isinstance(width, (int, np.integer)) or width <= 0
            for width in self.layer_sizes
        ):
            raise ValueError(
                "layer_sizes must contain at least two positive integer widths"
            )
        if activation not in {"tanh", "sin", "gelu", "relu", "linear"}:
            raise ValueError(f"Unsupported PINN activation: {activation!r}")
        self.activation = activation
        self.rng = np.random.default_rng(seed)

        # Initialize Xavier/Glorot weights
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []

        for din, dout in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            scale = np.sqrt(2.0 / (din + dout))
            w = self.rng.normal(0, scale, (din, dout)).astype(np.float64)
            b = np.zeros(dout, dtype=np.float64)
            self.weights.append(w)
            self.biases.append(b)

        # Adam state
        self._m_w = [np.zeros_like(w) for w in self.weights]
        self._v_w = [np.zeros_like(w) for w in self.weights]
        self._m_b = [np.zeros_like(b) for b in self.biases]
        self._v_b = [np.zeros_like(b) for b in self.biases]
        self._step_count = 0

    def _act(self, z: np.ndarray) -> np.ndarray:
        if self.activation == "tanh":
            return np.tanh(z)
        if self.activation == "sin":
            return np.sin(z)
        if self.activation == "relu":
            return np.maximum(0.0, z)
        if self.activation == "gelu":
            return (
                0.5
                * z
                * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (z + 0.044715 * (z**3))))
            )
        return z

    def _act_prime(self, z: np.ndarray, a: np.ndarray) -> np.ndarray:
        if self.activation == "tanh":
            return 1.0 - a**2
        if self.activation == "sin":
            return np.cos(z)
        if self.activation == "relu":
            return (z > 0).astype(np.float64)
        if self.activation == "gelu":
            s = np.sqrt(2.0 / np.pi) * (z + 0.044715 * z**3)
            tanh_s = np.tanh(s)
            sech2_s = 1.0 - tanh_s**2
            ds_dz = np.sqrt(2.0 / np.pi) * (1.0 + 3.0 * 0.044715 * z**2)
            return 0.5 * (1.0 + tanh_s) + 0.5 * z * sech2_s * ds_dz
        return np.ones_like(z)

    def forward(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
        """Forward pass caching intermediate pre-activations and activations."""
        a = np.asarray(X, dtype=np.float64)
        activations = [a]
        pre_acts = []

        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ w + b
            pre_acts.append(z)
            if i == len(self.weights) - 1:
                # Linear output layer
                a = z
            else:
                a = self._act(z)
            activations.append(a)

        return a, pre_acts, activations

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict output field for given coordinates."""
        out, _, _ = self.forward(X)
        return out

    def compute_gradient(
        self, X: np.ndarray, var_idx: int = 0, eps: float = 1e-4
    ) -> np.ndarray:
        """
        Compute directional gradient du/dx_i using central finite differences.
        """
        X = np.asarray(X, dtype=np.float64)
        X_plus = X.copy()
        X_minus = X.copy()
        X_plus[:, var_idx] += eps
        X_minus[:, var_idx] -= eps

        u_plus = self.predict(X_plus)
        u_minus = self.predict(X_minus)
        return (u_plus - u_minus) / (2.0 * eps)

    def compute_second_derivative(
        self, X: np.ndarray, var_idx: int = 0, eps: float = 1e-4
    ) -> np.ndarray:
        """
        Compute second directional derivative d^2u/dx_i^2 using central differences.
        """
        X = np.asarray(X, dtype=np.float64)
        u_center = self.predict(X)

        X_plus = X.copy()
        X_minus = X.copy()
        X_plus[:, var_idx] += eps
        X_minus[:, var_idx] -= eps

        u_plus = self.predict(X_plus)
        u_minus = self.predict(X_minus)
        return (u_plus - 2.0 * u_center + u_minus) / (eps**2)

    def burgers_residual(
        self, X_colloc: np.ndarray, nu: float = 0.01 / np.pi
    ) -> np.ndarray:
        """
        Burgers' equation residual: du/dt + u * du/dx - nu * d^2u/dx^2 = 0.
        Coordinates are assumed to be [x, t] (var_idx 0 is x, 1 is t).
        """
        u = self.predict(X_colloc)
        u_x = self.compute_gradient(X_colloc, var_idx=0)
        u_t = self.compute_gradient(X_colloc, var_idx=1)
        u_xx = self.compute_second_derivative(X_colloc, var_idx=0)
        return u_t + u * u_x - nu * u_xx

    def harmonic_oscillator_residual(
        self, t: np.ndarray, m: float = 1.0, k: float = 4.0, c: float = 0.4
    ) -> np.ndarray:
        """
        1D Damped Harmonic Oscillator: m * d^2u/dt^2 + c * du/dt + k * u = 0.
        """
        u = self.predict(t)
        u_t = self.compute_gradient(t, var_idx=0)
        u_tt = self.compute_second_derivative(t, var_idx=0)
        return m * u_tt + c * u_t + k * u

    def backward_data(
        self, X_data: np.ndarray, y_data: np.ndarray
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Backpropagation for mean squared error data/boundary loss."""
        X_data, y_data = self._prepare_data(X_data, y_data)
        preds, pre_acts, acts = self.forward(X_data)
        grad_out = (2.0 / preds.size) * (preds - y_data)

        grad_w = [np.zeros_like(w) for w in self.weights]
        grad_b = [np.zeros_like(b) for b in self.biases]

        delta = grad_out
        L = len(self.weights)

        for i in reversed(range(L)):
            a_prev = acts[i]
            grad_w[i] = a_prev.T @ delta
            grad_b[i] = np.sum(delta, axis=0)

            if i > 0:
                da = delta @ self.weights[i].T
                dz = self._act_prime(pre_acts[i - 1], acts[i])
                delta = da * dz

        return grad_w, grad_b

    def _prepare_data(self, X_data, y_data):
        X_data = np.asarray(X_data, dtype=np.float64)
        y_data = np.asarray(y_data, dtype=np.float64)
        if y_data.ndim == 1 and self.layer_sizes[-1] == 1:
            y_data = y_data[:, None]
        if (
            X_data.ndim != 2
            or X_data.shape[0] == 0
            or X_data.shape[1] != self.layer_sizes[0]
            or y_data.shape != (len(X_data), self.layer_sizes[-1])
        ):
            raise ValueError(
                "Expected nonempty X_data (samples, inputs) and y_data (samples, outputs)"
            )
        if not np.isfinite(X_data).all() or not np.isfinite(y_data).all():
            raise ValueError("PINN data must contain finite values")
        return X_data, y_data

    def train_step(
        self,
        X_data: np.ndarray,
        y_data: np.ndarray,
        X_colloc: np.ndarray | None = None,
        residual_fn: Callable[[PINN, np.ndarray], np.ndarray] | None = None,
        lambda_pde: float = 1.0,
        lr: float = 0.005,
    ) -> dict[str, float]:
        """
        Single training step minimizing total loss = L_data + lambda_pde * L_residual.
        """
        X_data, y_data = self._prepare_data(X_data, y_data)
        if (
            not np.isfinite(lr)
            or lr <= 0
            or not np.isfinite(lambda_pde)
            or lambda_pde < 0
        ):
            raise ValueError("lr must be positive and lambda_pde must be nonnegative")
        if (X_colloc is None) != (residual_fn is None):
            raise ValueError("Provide X_colloc and residual_fn together")
        # Data loss
        data_preds = self.predict(X_data)
        loss_data = float(np.mean((data_preds - y_data) ** 2))
        grad_w, grad_b = self.backward_data(X_data, y_data)

        loss_pde = 0.0
        if X_colloc is not None and residual_fn is not None and lambda_pde > 0:
            X_colloc = np.asarray(X_colloc, dtype=np.float64)
            if (
                X_colloc.ndim != 2
                or not len(X_colloc)
                or X_colloc.shape[1] != self.layer_sizes[0]
                or not np.isfinite(X_colloc).all()
            ):
                raise ValueError(
                    "X_colloc must be a nonempty finite (samples, inputs) array"
                )
            # Select the supported signature once, without swallowing errors
            # raised inside the user's callback.
            try:
                signature = inspect.signature(residual_fn)
            except (TypeError, ValueError):
                use_model_argument = False
            else:
                try:
                    signature.bind(X_colloc)
                    use_model_argument = False
                except TypeError:
                    signature.bind(self, X_colloc)
                    use_model_argument = True

            def _call_res(coords: np.ndarray) -> np.ndarray:
                value = (
                    residual_fn(self, coords)
                    if use_model_argument
                    else residual_fn(coords)
                )
                result = np.asarray(value, dtype=np.float64)
                if not result.size or not np.isfinite(result).all():
                    raise ValueError(
                        "Residual callback must return nonempty finite values"
                    )
                return result

            res = _call_res(X_colloc)
            loss_pde = float(np.mean(res**2))

            # Differentiate every parameter. Silent subsampling changes the
            # objective's gradient and formerly crashed for matrices >256 weights.
            eps_fd = 1e-4
            for parameters, gradients in (
                (self.weights, grad_w),
                (self.biases, grad_b),
            ):
                for parameter, gradient in zip(parameters, gradients):
                    for idx in np.ndindex(parameter.shape):
                        original = parameter[idx]
                        try:
                            parameter[idx] = original + eps_fd
                            plus = float(np.mean(_call_res(X_colloc) ** 2))
                            parameter[idx] = original - eps_fd
                            minus = float(np.mean(_call_res(X_colloc) ** 2))
                        finally:
                            parameter[idx] = original
                        gradient[idx] += lambda_pde * (plus - minus) / (2.0 * eps_fd)

        total_loss = loss_data + lambda_pde * loss_pde

        # Adam optimization update
        self._step_count += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        bc1 = 1.0 - beta1**self._step_count
        bc2 = 1.0 - beta2**self._step_count

        for i in range(len(self.weights)):
            # Weights
            self._m_w[i] = beta1 * self._m_w[i] + (1 - beta1) * grad_w[i]
            self._v_w[i] = beta2 * self._v_w[i] + (1 - beta2) * (grad_w[i] ** 2)
            m_hat = self._m_w[i] / bc1
            v_hat = self._v_w[i] / bc2
            self.weights[i] -= lr * m_hat / (np.sqrt(v_hat) + eps)

            # Biases
            self._m_b[i] = beta1 * self._m_b[i] + (1 - beta1) * grad_b[i]
            self._v_b[i] = beta2 * self._v_b[i] + (1 - beta2) * (grad_b[i] ** 2)
            mb_hat = self._m_b[i] / bc1
            vb_hat = self._v_b[i] / bc2
            self.biases[i] -= lr * mb_hat / (np.sqrt(vb_hat) + eps)

        return {
            "total_loss": total_loss,
            "loss_data": loss_data,
            "loss_pde": loss_pde,
        }

    def fit(
        self,
        X_data: np.ndarray,
        y_data: np.ndarray,
        X_colloc: np.ndarray | None = None,
        residual_fn: Callable[[PINN, np.ndarray], np.ndarray] | None = None,
        epochs: int = 100,
        lr: float = 0.01,
        lambda_pde: float = 1.0,
        verbose: bool = True,
    ) -> dict[str, list[float]]:
        """
        Train the PINN over a specified number of epochs.
        """
        history: dict[str, list[float]] = {
            "total_loss": [],
            "loss_data": [],
            "loss_pde": [],
        }

        for epoch in range(epochs):
            step_res = self.train_step(
                X_data=X_data,
                y_data=y_data,
                X_colloc=X_colloc,
                residual_fn=residual_fn,
                lambda_pde=lambda_pde,
                lr=lr,
            )
            for k, v in step_res.items():
                history[k].append(v)

            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                print(
                    f"Epoch {epoch + 1:4d}/{epochs:4d} | "
                    f"Total Loss: {step_res['total_loss']:.6f} | "
                    f"Data: {step_res['loss_data']:.6f} | "
                    f"PDE: {step_res['loss_pde']:.6f}"
                )

        return history
