"""
Built-in datasets for learning and experimentation.

Provides easy-to-load datasets with educational metadata
so beginners can start training models immediately.
"""

from typing import Optional, Tuple

import numpy as np


class Datasets:
    """
    Pre-loaded datasets for learning.

    Provides easy access to common datasets with educational metadata.

    Examples:
        >>> from neurogebra.datasets import Datasets
        >>> (X_train, y_train), (X_test, y_test) = Datasets.load_iris()
        >>> X, y = Datasets.load_simple_regression()
    """

    @staticmethod
    def load_mnist(
        flatten: bool = True,
        verbose: bool = True,
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
        Load MNIST handwritten digits dataset.

        This generates a synthetic version for learning purposes.
        For the real dataset, use ``keras.datasets.mnist``.

        Args:
            flatten: Flatten images to 1D (True for dense networks)
            verbose: Print dataset info

        Returns:
            ((X_train, y_train), (X_test, y_test))
        """
        if verbose:
            print("📦 Loading MNIST dataset...")
            print("   • Training samples: 60,000")
            print("   • Test samples: 10,000")
            print("   • Image size: 28x28 pixels")
            print("   • Classes: 10 (digits 0-9)")
            print()

        shape = (784,) if flatten else (28, 28)

        X_train = np.random.rand(60000, *shape).astype(np.float32)
        y_train = np.random.randint(0, 10, 60000)
        X_test = np.random.rand(10000, *shape).astype(np.float32)
        y_test = np.random.randint(0, 10, 10000)

        if verbose:
            print("✅ Dataset loaded!")

        return (X_train, y_train), (X_test, y_test)

    @staticmethod
    def load_iris(
        verbose: bool = True,
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
        Load Iris flower classification dataset.

        Perfect for beginners - small and simple.

        Args:
            verbose: Print dataset info

        Returns:
            ((X_train, y_train), (X_test, y_test))
        """
        if verbose:
            print("📦 Loading Iris dataset...")
            print("   • Total samples: 150")
            print("   • Features: 4 (sepal/petal measurements)")
            print("   • Classes: 3 (flower species)")
            print("   • Perfect for: First classification project")
            print()

        # Generate realistic-ish iris-like data
        np.random.seed(42)
        n_per_class = 50

        # Class 0: Setosa
        X0 = np.random.randn(n_per_class, 4) * 0.3 + [5.0, 3.4, 1.5, 0.2]
        # Class 1: Versicolor
        X1 = np.random.randn(n_per_class, 4) * 0.4 + [5.9, 2.8, 4.3, 1.3]
        # Class 2: Virginica
        X2 = np.random.randn(n_per_class, 4) * 0.4 + [6.6, 3.0, 5.6, 2.0]

        X = np.vstack([X0, X1, X2]).astype(np.float32)
        y = np.array(
            [0] * n_per_class + [1] * n_per_class + [2] * n_per_class
        )

        # Shuffle
        indices = np.random.permutation(150)
        X, y = X[indices], y[indices]

        split = 120
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # Reset random seed
        np.random.seed(None)

        if verbose:
            print("✅ Dataset loaded!")

        return (X_train, y_train), (X_test, y_test)

    @staticmethod
    def load_simple_regression(
        n_samples: int = 1000,
        noise: float = 0.5,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a simple regression dataset for learning.

        Generates data from y = 3x + 2 + noise.

        Args:
            n_samples: Number of samples
            noise: Standard deviation of Gaussian noise
            verbose: Print dataset info

        Returns:
            (X, y) tuple of numpy arrays
        """
        if verbose:
            print(
                f"📦 Generating regression dataset ({n_samples} samples)..."
            )
            print("   • Type: y = 3x + 2 + noise")
            print("   • Perfect for: First regression project")
            print()

        X = np.random.rand(n_samples, 1).astype(np.float32) * 10
        y = (3.0 * X + 2.0 + np.random.randn(n_samples, 1) * noise).astype(
            np.float32
        )

        if verbose:
            print("✅ Dataset ready!")

        return X, y.ravel()

    @staticmethod
    def load_xor(
        n_samples: int = 500,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate XOR dataset - classic non-linear classification problem.

        Demonstrates why we need non-linear activations.

        Args:
            n_samples: Number of samples
            verbose: Print dataset info

        Returns:
            (X, y) tuple
        """
        if verbose:
            print(f"📦 Generating XOR dataset ({n_samples} samples)...")
            print("   • Classic non-linear classification problem")
            print("   • Demonstrates need for hidden layers")
            print()

        X = np.random.rand(n_samples, 2).astype(np.float32)
        y = ((X[:, 0] > 0.5) ^ (X[:, 1] > 0.5)).astype(np.int32)

        if verbose:
            print("✅ Dataset ready!")

        return X, y

    @staticmethod
    def load_moons(
        n_samples: int = 500,
        noise: float = 0.1,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate two interleaving half-moons dataset.

        Good for testing non-linear decision boundaries.

        Args:
            n_samples: Total number of samples
            noise: Amount of noise
            verbose: Print dataset info

        Returns:
            (X, y) tuple
        """
        if verbose:
            print(f"📦 Generating moons dataset ({n_samples} samples)...")
            print("   • Two interleaving half-circles")
            print("   • Good for: Non-linear classification")
            print()

        n_half = n_samples // 2

        # Upper moon
        theta1 = np.linspace(0, np.pi, n_half)
        x1 = np.cos(theta1)
        y1 = np.sin(theta1)

        # Lower moon (shifted)
        theta2 = np.linspace(0, np.pi, n_samples - n_half)
        x2 = 1 - np.cos(theta2)
        y2 = 1 - np.sin(theta2) - 0.5

        X = np.vstack(
            [
                np.column_stack([x1, y1]),
                np.column_stack([x2, y2]),
            ]
        ).astype(np.float32)

        X += np.random.randn(*X.shape).astype(np.float32) * noise

        y = np.hstack(
            [np.zeros(n_half), np.ones(n_samples - n_half)]
        ).astype(np.int32)

        # Shuffle
        indices = np.random.permutation(n_samples)
        X, y = X[indices], y[indices]

        if verbose:
            print("✅ Dataset ready!")

        return X, y

    @staticmethod
    def load_circles(
        n_samples: int = 500,
        noise: float = 0.05,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate concentric circles dataset.

        Demonstrates radial decision boundaries.

        Args:
            n_samples: Number of samples
            noise: Amount of noise
            verbose: Print dataset info

        Returns:
            (X, y) tuple
        """
        if verbose:
            print(
                f"📦 Generating circles dataset ({n_samples} samples)..."
            )
            print("   • Two concentric circles")
            print("   • Shows radial decision boundaries")
            print()

        n_half = n_samples // 2

        # Outer circle
        theta1 = np.random.rand(n_half) * 2 * np.pi
        r1 = 1.0
        x1 = r1 * np.cos(theta1)
        y1 = r1 * np.sin(theta1)

        # Inner circle
        theta2 = np.random.rand(n_samples - n_half) * 2 * np.pi
        r2 = 0.5
        x2 = r2 * np.cos(theta2)
        y2 = r2 * np.sin(theta2)

        X = np.vstack(
            [
                np.column_stack([x1, y1]),
                np.column_stack([x2, y2]),
            ]
        ).astype(np.float32)

        X += np.random.randn(*X.shape).astype(np.float32) * noise

        y = np.hstack(
            [np.zeros(n_half), np.ones(n_samples - n_half)]
        ).astype(np.int32)

        indices = np.random.permutation(n_samples)
        X, y = X[indices], y[indices]

        if verbose:
            print("✅ Dataset ready!")

        return X, y
