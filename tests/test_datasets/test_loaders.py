"""Tests for Datasets loaders."""

import pytest
import numpy as np
from neurogebra.datasets.loaders import Datasets


class TestLoadMnist:
    """Test MNIST dataset loading."""

    def test_load_mnist_shape(self):
        """Test MNIST returns correct shapes."""
        (X_train, y_train), (X_test, y_test) = Datasets.load_mnist(
            flatten=True, verbose=False
        )
        assert X_train.shape == (60000, 784)
        assert y_train.shape == (60000,)
        assert X_test.shape == (10000, 784)
        assert y_test.shape == (10000,)

    def test_load_mnist_unflatten(self):
        """Test MNIST with flatten=False returns 2D images."""
        (X_train, _), _ = Datasets.load_mnist(
            flatten=False, verbose=False
        )
        assert X_train.shape == (60000, 28, 28)

    def test_load_mnist_dtype(self):
        """Test MNIST data type."""
        (X_train, y_train), _ = Datasets.load_mnist(verbose=False)
        assert X_train.dtype == np.float32

    def test_load_mnist_labels_range(self):
        """Test MNIST labels are in [0, 9]."""
        _, (_, y_test) = Datasets.load_mnist(verbose=False)
        assert y_test.min() >= 0
        assert y_test.max() <= 9

    def test_load_mnist_verbose(self, capsys):
        """Test verbose mode prints dataset info."""
        Datasets.load_mnist(verbose=True)
        captured = capsys.readouterr()
        assert "MNIST" in captured.out
        assert "loaded" in captured.out.lower()


class TestLoadIris:
    """Test Iris dataset loading."""

    def test_load_iris_shape(self):
        """Test Iris returns correct shapes."""
        (X_train, y_train), (X_test, y_test) = Datasets.load_iris(
            verbose=False
        )
        assert X_train.shape == (120, 4)
        assert y_train.shape == (120,)
        assert X_test.shape == (30, 4)
        assert y_test.shape == (30,)

    def test_load_iris_dtype(self):
        """Test Iris data type."""
        (X_train, _), _ = Datasets.load_iris(verbose=False)
        assert X_train.dtype == np.float32

    def test_load_iris_classes(self):
        """Test Iris has 3 classes."""
        (_, y_train), (_, y_test) = Datasets.load_iris(verbose=False)
        all_labels = np.concatenate([y_train, y_test])
        assert set(all_labels.tolist()) == {0, 1, 2}

    def test_load_iris_deterministic(self):
        """Test Iris returns same data on repeated calls."""
        (X1, _), _ = Datasets.load_iris(verbose=False)
        (X2, _), _ = Datasets.load_iris(verbose=False)
        np.testing.assert_array_equal(X1, X2)

    def test_load_iris_verbose(self, capsys):
        """Test verbose mode prints dataset info."""
        Datasets.load_iris(verbose=True)
        captured = capsys.readouterr()
        assert "Iris" in captured.out


class TestLoadSimpleRegression:
    """Test simple regression dataset."""

    def test_load_shape(self):
        """Test regression dataset shapes."""
        X, y = Datasets.load_simple_regression(
            n_samples=500, verbose=False
        )
        assert X.shape == (500, 1)
        assert y.shape == (500,)

    def test_load_dtype(self):
        """Test data types."""
        X, y = Datasets.load_simple_regression(verbose=False)
        assert X.dtype == np.float32
        assert y.dtype == np.float32

    def test_load_custom_samples(self):
        """Test custom number of samples."""
        X, y = Datasets.load_simple_regression(
            n_samples=200, verbose=False
        )
        assert len(X) == 200
        assert len(y) == 200

    def test_load_verbose(self, capsys):
        """Test verbose output."""
        Datasets.load_simple_regression(verbose=True)
        captured = capsys.readouterr()
        assert "regression" in captured.out.lower()


class TestLoadXor:
    """Test XOR dataset."""

    def test_load_shape(self):
        """Test XOR dataset shapes."""
        X, y = Datasets.load_xor(n_samples=100, verbose=False)
        assert X.shape == (100, 2)
        assert y.shape == (100,)

    def test_load_binary_labels(self):
        """Test XOR has binary labels."""
        X, y = Datasets.load_xor(verbose=False)
        assert set(y.tolist()).issubset({0, 1})

    def test_load_dtype(self):
        """Test data types."""
        X, y = Datasets.load_xor(verbose=False)
        assert X.dtype == np.float32
        assert y.dtype == np.int32


class TestLoadMoons:
    """Test moons dataset."""

    def test_load_shape(self):
        """Test moons dataset shapes."""
        X, y = Datasets.load_moons(n_samples=200, verbose=False)
        assert X.shape == (200, 2)
        assert y.shape == (200,)

    def test_load_binary_labels(self):
        """Test moons has binary labels."""
        X, y = Datasets.load_moons(verbose=False)
        assert set(y.tolist()).issubset({0, 1})

    def test_load_with_noise(self):
        """Test loading with different noise levels."""
        X1, _ = Datasets.load_moons(noise=0.0, verbose=False)
        X2, _ = Datasets.load_moons(noise=0.5, verbose=False)
        # Higher noise should produce more spread
        assert X2.std() >= X1.std() * 0.5  # rough check


class TestLoadCircles:
    """Test circles dataset."""

    def test_load_shape(self):
        """Test circles dataset shapes."""
        X, y = Datasets.load_circles(n_samples=300, verbose=False)
        assert X.shape == (300, 2)
        assert y.shape == (300,)

    def test_load_binary_labels(self):
        """Test circles has binary labels."""
        X, y = Datasets.load_circles(verbose=False)
        assert set(y.tolist()).issubset({0, 1})

    def test_load_dtype(self):
        """Test data types."""
        X, y = Datasets.load_circles(verbose=False)
        assert X.dtype == np.float32
        assert y.dtype == np.int32

    def test_load_verbose(self, capsys):
        """Test verbose output."""
        Datasets.load_circles(verbose=True)
        captured = capsys.readouterr()
        assert "circles" in captured.out.lower()
