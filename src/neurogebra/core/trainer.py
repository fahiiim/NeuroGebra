"""
Training utilities for trainable expressions.

Implements optimization algorithms for learning parameters in expressions.
"""

from typing import Callable, Optional

import numpy as np

from neurogebra.core.expression import Expression


class Trainer:
    """
    Trainer for optimizing expression parameters.

    Supports various optimization algorithms for fitting expressions to data.

    Examples:
        >>> from neurogebra.core.expression import Expression
        >>> expr = Expression("linear", "a*x + b",
        ...                   params={"a": 0.0, "b": 0.0},
        ...                   trainable_params=["a", "b"])
        >>> trainer = Trainer(expr, learning_rate=0.01)
        >>> X = np.array([1, 2, 3, 4, 5])
        >>> y = np.array([3, 5, 7, 9, 11])  # y = 2x + 1
        >>> history = trainer.fit(X, y, epochs=100)
    """

    def __init__(
        self,
        expression: Expression,
        learning_rate: float = 0.01,
        optimizer: str = "sgd",
    ):
        """
        Initialize Trainer.

        Args:
            expression: Expression to train
            learning_rate: Learning rate for optimization
            optimizer: Optimization algorithm ('sgd', 'adam')
        """
        self.expression = expression
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.history: dict[str, list] = {"loss": [], "params": []}

        # Adam optimizer state
        self._m: dict[str, float] = {}
        self._v: dict[str, float] = {}
        self._t = 0

        for param_name in self.expression.trainable_params:
            self._m[param_name] = 0.0
            self._v[param_name] = 0.0

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        batch_size: Optional[int] = None,
        loss_fn: str = "mse",
        verbose: bool = True,
        callback: Optional[Callable] = None,
    ) -> dict[str, list]:
        """
        Fit expression to data.

        Args:
            X: Input data (N,) or (N, D)
            y: Target data (N,)
            epochs: Number of training epochs
            batch_size: Mini-batch size (None = full batch)
            loss_fn: Loss function ('mse', 'mae')
            verbose: Print training progress
            callback: Optional callback function called each epoch

        Returns:
            Training history with loss and parameter values per epoch
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if batch_size is None:
            batch_size = len(X)

        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(len(X))
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = 0.0
            n_batches = 0

            # Mini-batch training
            for i in range(0, len(X), batch_size):
                X_batch = X_shuffled[i : i + batch_size]
                y_batch = y_shuffled[i : i + batch_size]

                # Forward pass
                predictions = self._forward(X_batch)

                # Compute loss
                loss = self._compute_loss(predictions, y_batch, loss_fn)
                epoch_loss += loss
                n_batches += 1

                # Backward pass (numerical gradients for MVP)
                self._backward(X_batch, y_batch, loss_fn)

            # Record history
            avg_loss = epoch_loss / n_batches
            self.history["loss"].append(avg_loss)
            self.history["params"].append(self.expression.params.copy())

            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch:>4d}/{epochs}: Loss = {avg_loss:.6f}")

            if callback is not None:
                callback(epoch, avg_loss, self.expression.params.copy())

        return self.history

    def _forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass through expression using vectorized evaluation."""
        try:
            preds = self.expression.eval(x=X)
            preds_arr = np.asarray(preds, dtype=np.float64)
            if preds_arr.shape == () and len(X) > 0:
                return np.full(len(X), float(preds_arr), dtype=np.float64)
            # If 2D with single column and inputs are 1D, flatten to 1D
            if preds_arr.ndim == 2 and preds_arr.shape[1] == 1:
                preds_arr = preds_arr.ravel()
            return preds_arr
        except Exception:
            # Fallback to sample-by-sample evaluation if expression requires scalar inputs
            return np.array(
                [float(self.expression.eval(x=float(x))) for x in X], dtype=np.float64
            )

    def _compute_loss(
        self, predictions: np.ndarray, targets: np.ndarray, loss_fn: str
    ) -> float:
        """Compute loss value."""
        p = np.asarray(predictions, dtype=np.float64)
        t = np.asarray(targets, dtype=np.float64)
        if p.shape != t.shape:
            if p.size == t.size:
                p = p.reshape(t.shape)
            else:
                raise ValueError(f"Shape mismatch: predictions {p.shape} vs targets {t.shape}")

        if loss_fn == "mse":
            return float(np.mean((p - t) ** 2))
        elif loss_fn == "mae":
            return float(np.mean(np.abs(p - t)))
        elif loss_fn == "huber":
            delta = 1.0
            diff = np.abs(p - t)
            return float(
                np.mean(
                    np.where(
                        diff <= delta,
                        0.5 * diff**2,
                        delta * diff - 0.5 * delta**2,
                    )
                )
            )
        else:
            raise ValueError(f"Unknown loss function: {loss_fn}")

    def _backward(self, X: np.ndarray, y: np.ndarray, loss_fn: str):
        """
        Backward pass using analytical gradients when possible,
        falling back to central finite differences.
        """
        if self.optimizer == "adam":
            self._t += 1

        preds = self._forward(X)
        y = np.asarray(y, dtype=np.float64)
        if preds.shape != y.shape and preds.size == y.size:
            preds = preds.reshape(y.shape)
        n = len(X)

        # Phase 1: Compute gradients for all trainable parameters from snapshot
        gradients = {}
        original_params = self.expression.params.copy()

        for param_name in self.expression.trainable_params:
            if param_name not in self.expression.params:
                continue

            gradient = None
            # Attempt analytical gradient via symbolic differentiation
            try:
                grad_expr = self.expression.gradient(param_name)
                df_dp = grad_expr.eval(x=X)
                df_dp = np.asarray(df_dp, dtype=np.float64)
                if df_dp.shape == () and n > 0:
                    df_dp = np.full_like(preds, float(df_dp), dtype=np.float64)
                elif df_dp.shape != preds.shape and df_dp.size == preds.size:
                    df_dp = df_dp.reshape(preds.shape)

                if loss_fn == "mse":
                    dL_df = 2.0 * (preds - y) / n
                    gradient = float(np.sum(dL_df * df_dp))
                elif loss_fn == "mae":
                    dL_df = np.sign(preds - y) / n
                    gradient = float(np.sum(dL_df * df_dp))
                elif loss_fn == "huber":
                    delta = 1.0
                    diff = preds - y
                    dL_df = (
                        np.where(np.abs(diff) <= delta, diff, delta * np.sign(diff)) / n
                    )
                    gradient = float(np.sum(dL_df * df_dp))
            except Exception:
                gradient = None

            # Fallback to central finite differences if analytical gradient not computed
            if gradient is None or not np.isfinite(gradient):
                epsilon = 1e-5
                original_value = original_params[param_name]

                try:
                    self.expression.params[param_name] = original_value + epsilon
                    pred_plus = self._forward(X)
                    loss_plus = self._compute_loss(pred_plus, y, loss_fn)

                    self.expression.params[param_name] = original_value - epsilon
                    pred_minus = self._forward(X)
                    loss_minus = self._compute_loss(pred_minus, y, loss_fn)

                    gradient = (loss_plus - loss_minus) / (2 * epsilon)
                finally:
                    self.expression.params[param_name] = original_value

            gradients[param_name] = gradient

        # Phase 2: Simultaneous parameter update
        for param_name, gradient in gradients.items():
            if gradient is None or not np.isfinite(gradient):
                continue
            original_value = original_params[param_name]
            if self.optimizer == "adam":
                self._adam_update(param_name, gradient)
            else:
                # SGD
                self.expression.params[param_name] = (
                    original_value - self.learning_rate * gradient
                )

    def _adam_update(
        self,
        param_name: str,
        gradient: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        """Adam optimizer update step."""
        t = max(1, self._t)
        self._m[param_name] = beta1 * self._m[param_name] + (1 - beta1) * gradient
        self._v[param_name] = beta2 * self._v[param_name] + (1 - beta2) * gradient**2

        m_hat = self._m[param_name] / (1 - beta1**t)
        v_hat = self._v[param_name] / (1 - beta2**t)

        self.expression.params[param_name] -= (
            self.learning_rate * m_hat / (np.sqrt(v_hat) + eps)
        )

    def reset(self):
        """Reset trainer state."""
        self.history = {"loss": [], "params": []}
        self._t = 0
        for param_name in self.expression.trainable_params:
            self._m[param_name] = 0.0
            self._v[param_name] = 0.0
