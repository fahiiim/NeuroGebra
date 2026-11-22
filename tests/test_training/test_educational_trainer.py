"""Tests for EducationalTrainer."""

import numpy as np

from neurogebra.training.educational_trainer import EducationalTrainer


class FakeModel:
    """Minimal model stub for trainer tests."""

    def __init__(self):
        self.name = "FakeModel"
        self.loss_name = "mse"
        self.optimizer = "adam"
        self.learning_rate = 0.001
        self.layers = []
        self.history = {
            "loss": [],
            "accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }


class TestEducationalTrainerInit:
    """Test EducationalTrainer initialization."""

    def test_creates_successfully(self):
        """Test trainer creates successfully."""
        model = FakeModel()
        trainer = EducationalTrainer(model, verbose=False)
        assert trainer is not None
        assert trainer.model is model

    def test_default_settings(self):
        """Test default settings."""
        model = FakeModel()
        trainer = EducationalTrainer(model, verbose=False)
        assert trainer.verbose is False
        assert trainer.visualize is False
        assert trainer.explain_steps is True

    def test_history_initialized(self):
        """Test history dict is initialized."""
        model = FakeModel()
        trainer = EducationalTrainer(model, verbose=False)
        assert "loss" in trainer.history
        assert "accuracy" in trainer.history
        assert "val_loss" in trainer.history
        assert "val_accuracy" in trainer.history


class TestEducationalTrainerTrain:
    """Test the training loop."""

    def test_train_rejects_stub_without_simulate_flag(self):
        """Test trainer raises TypeError if model lacks real forward/backward and simulate=False."""
        import pytest

        model = FakeModel()
        trainer = EducationalTrainer(
            model, verbose=False, explain_steps=False, simulate=False
        )
        X = np.random.rand(20, 4)
        y = np.random.rand(20, 1)
        with pytest.raises(TypeError, match="Model does not implement required"):
            trainer.train(X, y, epochs=1, batch_size=4)

    def test_train_returns_history_simulated(self):
        """Test train returns complete history in simulation mode."""
        model = FakeModel()
        trainer = EducationalTrainer(
            model, verbose=False, explain_steps=False, simulate=True
        )
        X = np.random.rand(100, 10)
        y = np.random.randint(0, 2, 100)
        history = trainer.train(X, y, epochs=5, batch_size=32, validation_split=0.2)

        assert isinstance(history, dict)
        assert history.get("is_simulated") is True
        assert len(history["loss"]) == 5
        assert len(history["accuracy"]) == 5
        assert len(history["val_loss"]) == 5
        assert len(history["val_accuracy"]) == 5

    def test_loss_decreases_overall(self):
        """Test that simulated loss generally decreases."""
        model = FakeModel()
        trainer = EducationalTrainer(
            model, verbose=False, explain_steps=False, simulate=True
        )
        X = np.random.rand(200, 10)
        y = np.random.randint(0, 2, 200)
        history = trainer.train(X, y, epochs=20, batch_size=32, validation_split=0.2)

        # First loss should be higher than last on average
        assert history["loss"][0] > history["loss"][-1]

    def test_accuracy_increases_overall(self):
        """Test that simulated accuracy generally increases."""
        model = FakeModel()
        trainer = EducationalTrainer(
            model, verbose=False, explain_steps=False, simulate=True
        )
        X = np.random.rand(200, 10)
        y = np.random.randint(0, 2, 200)
        history = trainer.train(X, y, epochs=20, batch_size=32, validation_split=0.2)

        assert history["accuracy"][-1] > history["accuracy"][0]

    def test_verbose_prints_progress(self, capsys):
        """Test verbose mode prints epoch progress."""
        model = FakeModel()
        trainer = EducationalTrainer(
            model, verbose=True, explain_steps=False, simulate=True
        )
        X = np.random.rand(100, 10)
        y = np.random.randint(0, 2, 100)
        trainer.train(X, y, epochs=3, batch_size=32, validation_split=0.2)
        captured = capsys.readouterr()
        assert "Epoch" in captured.out

    def test_explain_steps_prints_explanation(self, capsys):
        """Test explain_steps prints what's happening."""
        model = FakeModel()
        trainer = EducationalTrainer(
            model, verbose=False, explain_steps=True, simulate=True
        )
        X = np.random.rand(100, 10)
        y = np.random.randint(0, 2, 100)
        trainer.train(X, y, epochs=3, batch_size=32, validation_split=0.2)
        captured = capsys.readouterr()
        assert (
            "about to happen" in captured.out.lower()
            or "training" in captured.out.lower()
        )

    def test_validation_split(self):
        """Test data is split correctly for validation."""
        model = FakeModel()
        trainer = EducationalTrainer(
            model, verbose=False, explain_steps=False, simulate=True
        )
        X = np.random.rand(100, 10)
        y = np.random.randint(0, 2, 100)
        history = trainer.train(X, y, epochs=2, batch_size=32, validation_split=0.3)
        # Should complete without error
        assert len(history["val_loss"]) == 2

    def test_small_dataset(self):
        """Test training with very small dataset."""
        model = FakeModel()
        trainer = EducationalTrainer(
            model, verbose=False, explain_steps=False, simulate=True
        )
        X = np.random.rand(10, 5)
        y = np.random.randint(0, 2, 10)
        history = trainer.train(X, y, epochs=2, batch_size=4, validation_split=0.2)
        assert len(history["loss"]) == 2

    def test_single_epoch(self):
        """Test training for a single epoch."""
        model = FakeModel()
        trainer = EducationalTrainer(
            model, verbose=False, explain_steps=False, simulate=True
        )
        X = np.random.rand(50, 10)
        y = np.random.randint(0, 2, 50)
        history = trainer.train(X, y, epochs=1, batch_size=16, validation_split=0.2)
        assert len(history["loss"]) == 1

    def test_real_model_training_deterministic(self):
        """Test that EducationalTrainer trains an authentic Model and decreases loss."""
        from neurogebra.builders.model_builder import ModelBuilder

        builder = ModelBuilder()
        model = builder.Sequential(
            [
                builder.Dense(4, activation="relu"),
                builder.Dense(1),
            ]
        )
        model.compile(loss="mse", optimizer="adam", learning_rate=0.05)
        trainer = EducationalTrainer(
            model, verbose=False, explain_steps=False, simulate=False
        )

        np.random.seed(42)
        X = np.random.randn(40, 4)
        W_true = np.array([[1.0], [-2.0], [0.5], [1.5]])
        y = X @ W_true

        history = trainer.train(X, y, epochs=15, batch_size=8, validation_split=0.2)
        assert history.get("is_simulated") is not True
        assert history["loss"][-1] < history["loss"][0]
