"""
Training utilities for trainable expressions.

Implements optimization algorithms for learning parameters in expressions.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple
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
        self.history: Dict[str, List] = {"loss": [], "params": []}

        # Adam optimizer state
        self._m: Dict[str, float] = {}
        self._v: Dict[str, float] = {}
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
    ) -> Dict[str, List]:
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
        """Forward pass through expression."""
        results = []
        for x in X:
            result = self.expression.eval(x=float(x))
            results.append(float(result))
        return np.array(results)

    def _compute_loss(
        self, predictions: np.ndarray, targets: np.ndarray, loss_fn: str
    ) -> float:
        """Compute loss value."""
        if loss_fn == "mse":
            return float(np.mean((predictions - targets) ** 2))
        elif loss_fn == "mae":
            return float(np.mean(np.abs(predictions - targets)))
        elif loss_fn == "huber":
            delta = 1.0
            diff = np.abs(predictions - targets)
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
        Backward pass using numerical gradients.

        For MVP, use finite differences. Later versions can use
        symbolic differentiation or autograd.
        """
        epsilon = 1e-5

        for param_name in self.expression.trainable_params:
            if param_name not in self.expression.params:
                continue

            original_value = self.expression.params[param_name]

            # Compute gradient via central finite difference
            self.expression.params[param_name] = original_value + epsilon
            pred_plus = self._forward(X)
            loss_plus = self._compute_loss(pred_plus, y, loss_fn)

            self.expression.params[param_name] = original_value - epsilon
            pred_minus = self._forward(X)
            loss_minus = self._compute_loss(pred_minus, y, loss_fn)

            # Restore original
            self.expression.params[param_name] = original_value

            # Gradient
            gradient = (loss_plus - loss_minus) / (2 * epsilon)

            # Update parameter based on optimizer
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
        self._t += 1

        self._m[param_name] = beta1 * self._m[param_name] + (1 - beta1) * gradient
        self._v[param_name] = beta2 * self._v[param_name] + (1 - beta2) * gradient**2

        m_hat = self._m[param_name] / (1 - beta1**self._t)
        v_hat = self._v[param_name] / (1 - beta2**self._t)

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
