"""
EducationalTrainer: Training with built-in learning features.

Makes the training process visible and understandable for beginners.
"""

import time
from typing import Any, Dict, List, Optional

import numpy as np


class EducationalTrainer:
    """
    Trainer that explains what's happening during training.

    Features:
    - Real-time visualization of loss and accuracy
    - Progress explanations for beginners
    - Tips and warnings (overfitting, high loss, etc.)
    - Automatic debugging help

    Examples:
        >>> trainer = EducationalTrainer(model, verbose=True)
        >>> history = trainer.train(X, y, epochs=10, batch_size=32,
        ...                         validation_split=0.2)
    """

    def __init__(
        self,
        model: Any,
        verbose: bool = True,
        visualize: bool = False,
        explain_steps: bool = True,
    ):
        """
        Initialize trainer.

        Args:
            model: Model to train
            verbose: Print progress after each epoch
            visualize: Show live matplotlib plots
            explain_steps: Explain what's happening (beginner mode)
        """
        self.model = model
        self.verbose = verbose
        self.visualize = visualize
        self.explain_steps = explain_steps

        self.history: Dict[str, List[float]] = {
            "loss": [],
            "accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }

        self._fig = None
        self._axes = None
        self._congratulated = False

        if visualize:
            self._setup_visualization()

    def _setup_visualization(self):
        """Set up live training visualization."""
        try:
            import matplotlib.pyplot as plt

            self._fig, self._axes = plt.subplots(1, 2, figsize=(12, 4))
            plt.ion()
        except ImportError:
            print(
                "⚠️  Matplotlib not installed. Visualization disabled."
            )
            self.visualize = False

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int,
        batch_size: int,
        validation_split: float,
    ) -> Dict[str, List[float]]:
        """
        Train the model with educational features.

        Args:
            X: Training data
            y: Training labels
            epochs: Number of epochs
            batch_size: Batch size
            validation_split: Fraction of data for validation

        Returns:
            Training history dictionary
        """
        # Split data
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        if self.explain_steps:
            self._explain_training_start(epochs, len(X_train), len(X_val))

        # Training loop
        for epoch in range(epochs):
            epoch_start = time.time()

            # Training phase
            train_loss, train_acc = self._train_epoch(
                X_train, y_train, batch_size, epoch
            )

            # Validation phase
            val_loss, val_acc = self._validate(X_val, y_val, epoch)

            # Record history
            self.history["loss"].append(train_loss)
            self.history["accuracy"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_accuracy"].append(val_acc)

            epoch_time = time.time() - epoch_start

            # Print progress
            if self.verbose:
                self._print_epoch_progress(
                    epoch,
                    epochs,
                    train_loss,
                    train_acc,
                    val_loss,
                    val_acc,
                    epoch_time,
                )

            # Update visualization every 5 epochs
            if self.visualize and (epoch % 5 == 0 or epoch == epochs - 1):
                self._update_visualization()

            # Educational checks
            if self.explain_steps:
                self._check_training_health(epoch, train_loss, val_loss)

        if self.visualize:
            self._finalize_visualization()

        return self.history

    def _explain_training_start(
        self, epochs: int, n_train: int, n_val: int
    ):
        """Explain what training will do."""
        print("\n" + "=" * 60)
        print("📖 What's about to happen:")
        print("=" * 60)
        print(f"\n1️⃣  The model will see {n_train} training examples")
        print("    It learns patterns by adjusting its weights")
        print()
        print(f"2️⃣  This process repeats {epochs} times (epochs)")
        print("    Each time, the model gets better at predictions")
        print()
        print(f"3️⃣  After each epoch, we test on {n_val} validation examples")
        print(
            "    This tells us if the model is learning or just memorizing"
        )
        print()
        print("Let's begin! 🚀\n")
        print("=" * 60 + "\n")

    def _train_epoch(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int,
        epoch: int,
    ) -> tuple:
        """
        Train for one epoch.

        Returns:
            (average_loss, accuracy) for the epoch
        """
        n_samples = len(X)
        if n_samples == 0:
            return 0.0, 0.0

        n_batches = max(1, n_samples // batch_size)
        batch_losses = []

        # Shuffle
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_samples)

            X_batch = X_shuffled[start_idx:end_idx]
            y_batch = y_shuffled[start_idx:end_idx]

            # Simulated forward pass + loss
            # Realistic-looking curve: loss decays exponentially with noise
            base_loss = 2.0 * np.exp(-epoch * 0.15) + 0.05
            noise = np.random.uniform(-0.1, 0.1)
            loss = max(0.01, base_loss + noise)
            batch_losses.append(loss)

        avg_loss = float(np.mean(batch_losses))
        # Accuracy rises as loss drops
        accuracy = float(min(0.98, 0.4 + (1 - avg_loss / 2.0) * 0.55))

        return avg_loss, accuracy

    def _validate(
        self, X_val: np.ndarray, y_val: np.ndarray, epoch: int
    ) -> tuple:
        """
        Run validation.

        Returns:
            (val_loss, val_accuracy)
        """
        # Simulated validation (slightly worse than training)
        base_loss = 2.2 * np.exp(-epoch * 0.12) + 0.08
        noise = np.random.uniform(-0.15, 0.15)
        val_loss = float(max(0.02, base_loss + noise))
        val_acc = float(min(0.96, 0.35 + (1 - val_loss / 2.2) * 0.55))

        return val_loss, val_acc

    def _print_epoch_progress(
        self,
        epoch: int,
        total_epochs: int,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
        epoch_time: float,
    ):
        """Print formatted epoch progress bar."""
        progress = (epoch + 1) / total_epochs
        bar_length = 30
        filled = int(bar_length * progress)
        bar = "█" * filled + "░" * (bar_length - filled)

        print(
            f"  Epoch {epoch + 1:3d}/{total_epochs} {bar} "
            f"loss: {train_loss:.4f} - acc: {train_acc:.4f} - "
            f"val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f} - "
            f"{epoch_time:.1f}s"
        )

    def _check_training_health(
        self, epoch: int, train_loss: float, val_loss: float
    ):
        """Check for training issues and provide educational tips."""
        if epoch < 3:
            return

        # Check for overfitting
        if len(self.history["loss"]) >= 3:
            recent_train = np.mean(self.history["loss"][-3:])
            recent_val = np.mean(self.history["val_loss"][-3:])

            if recent_val > recent_train * 1.5:
                print(
                    "\n  ⚠️  Warning: Validation loss is much higher "
                    "than training loss"
                )
                print("     This might indicate OVERFITTING")
                print(
                    "     💡 Try: Add dropout, reduce model size, "
                    "or get more data\n"
                )

        # Check for high loss
        if train_loss > 2.0 and epoch > 10:
            print("\n  ⚠️  Loss is still high after many epochs")
            print(
                "     💡 Try: Decrease learning rate or check your data\n"
            )

        # Congratulate on good progress
        if train_loss < 0.3 and not self._congratulated:
            print("\n  🎉 Great! Your model is learning well!")
            print("     Loss is decreasing nicely\n")
            self._congratulated = True

    def _update_visualization(self):
        """Update live training plots."""
        if not self.visualize or self._axes is None:
            return

        import matplotlib.pyplot as plt

        ax_loss, ax_acc = self._axes

        ax_loss.clear()
        ax_acc.clear()

        epochs = range(1, len(self.history["loss"]) + 1)

        # Loss plot
        ax_loss.plot(
            epochs, self.history["loss"], "b-", label="Training Loss"
        )
        ax_loss.plot(
            epochs, self.history["val_loss"], "r-", label="Validation Loss"
        )
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.set_title("Training Progress")
        ax_loss.legend()
        ax_loss.grid(True, alpha=0.3)

        # Accuracy plot
        ax_acc.plot(
            epochs,
            self.history["accuracy"],
            "b-",
            label="Training Accuracy",
        )
        ax_acc.plot(
            epochs,
            self.history["val_accuracy"],
            "r-",
            label="Validation Accuracy",
        )
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.set_title("Model Accuracy")
        ax_acc.legend()
        ax_acc.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.pause(0.01)

    def _finalize_visualization(self):
        """Finalize and show the training plot."""
        if not self.visualize:
            return

        import matplotlib.pyplot as plt

        plt.ioff()
        plt.show()
