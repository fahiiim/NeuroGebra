"""Tests for Trainer module."""

import pytest
import numpy as np
from neurogebra.core.expression import Expression
from neurogebra.core.trainer import Trainer


class TestTrainerInit:
    """Test trainer initialization."""

    def test_basic_init(self):
        """Test basic trainer creation."""
        expr = Expression(
            "linear",
            "m*x + b",
            params={"m": 0.0, "b": 0.0},
            trainable_params=["m", "b"],
        )
        trainer = Trainer(expr, learning_rate=0.01)
        assert trainer.learning_rate == 0.01
        assert trainer.optimizer == "sgd"

    def test_adam_init(self):
        """Test Adam optimizer initialization."""
        expr = Expression(
            "linear",
            "m*x + b",
            params={"m": 0.0, "b": 0.0},
            trainable_params=["m", "b"],
        )
        trainer = Trainer(expr, optimizer="adam")
        assert trainer.optimizer == "adam"


class TestTrainerFit:
    """Test training functionality."""

    def test_linear_fit(self):
        """Test fitting a linear expression to linear data."""
        np.random.seed(42)

        expr = Expression(
            "linear",
            "m*x + b",
            params={"m": 0.0, "b": 0.0},
            trainable_params=["m", "b"],
        )
        trainer = Trainer(expr, learning_rate=0.01)

        # y = 2x + 1
        X = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = 2 * X + 1

        history = trainer.fit(X, y, epochs=200, verbose=False)

        # Check that loss decreased
        assert history["loss"][-1] < history["loss"][0]

        # Check learned parameters are close
        assert abs(expr.params["m"] - 2.0) < 0.5
        assert abs(expr.params["b"] - 1.0) < 1.0

    def test_loss_decreases(self):
        """Test that loss monotonically decreases (mostly)."""
        np.random.seed(42)

        expr = Expression(
            "linear",
            "m*x + b",
            params={"m": 0.0, "b": 0.0},
            trainable_params=["m", "b"],
        )
        trainer = Trainer(expr, learning_rate=0.001)

        X = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = 2 * X + 1

        history = trainer.fit(X, y, epochs=50, verbose=False)

        # Loss at end should be less than loss at start
        assert history["loss"][-1] < history["loss"][0]

    def test_mae_loss(self):
        """Test training with MAE loss."""
        np.random.seed(42)

        expr = Expression(
            "linear",
            "m*x + b",
            params={"m": 0.0, "b": 0.0},
            trainable_params=["m", "b"],
        )
        trainer = Trainer(expr, learning_rate=0.01)

        X = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = 2 * X + 1

        history = trainer.fit(X, y, epochs=50, loss_fn="mae", verbose=False)
        assert history["loss"][-1] < history["loss"][0]

    def test_invalid_loss_raises(self):
        """Test that invalid loss function raises error."""
        expr = Expression(
            "linear",
            "m*x + b",
            params={"m": 0.0, "b": 0.0},
            trainable_params=["m", "b"],
        )
        trainer = Trainer(expr, learning_rate=0.01)

        X = np.array([1.0, 2.0])
        y = np.array([2.0, 4.0])

        with pytest.raises(ValueError, match="Unknown loss function"):
            trainer.fit(X, y, epochs=1, loss_fn="invalid", verbose=False)

    def test_history_tracking(self):
        """Test that history is properly tracked."""
        expr = Expression(
            "linear",
            "m*x + b",
            params={"m": 0.0, "b": 0.0},
            trainable_params=["m", "b"],
        )
        trainer = Trainer(expr, learning_rate=0.01)

        X = np.array([1.0, 2.0, 3.0])
        y = np.array([2.0, 4.0, 6.0])

        history = trainer.fit(X, y, epochs=10, verbose=False)
        assert len(history["loss"]) == 10
        assert len(history["params"]) == 10

    def test_callback(self):
        """Test callback function."""
        callback_calls = []

        def my_callback(epoch, loss, params):
            callback_calls.append(epoch)

        expr = Expression(
            "linear",
            "m*x + b",
            params={"m": 0.0, "b": 0.0},
            trainable_params=["m", "b"],
        )
        trainer = Trainer(expr, learning_rate=0.01)

        X = np.array([1.0, 2.0, 3.0])
        y = np.array([2.0, 4.0, 6.0])

        trainer.fit(X, y, epochs=5, verbose=False, callback=my_callback)
        assert len(callback_calls) == 5

    def test_reset(self):
        """Test trainer reset."""
        expr = Expression(
            "linear",
            "m*x + b",
            params={"m": 0.0, "b": 0.0},
            trainable_params=["m", "b"],
        )
        trainer = Trainer(expr, learning_rate=0.01)

        X = np.array([1.0, 2.0, 3.0])
        y = np.array([2.0, 4.0, 6.0])

        trainer.fit(X, y, epochs=5, verbose=False)
        assert len(trainer.history["loss"]) == 5

        trainer.reset()
        assert len(trainer.history["loss"]) == 0
