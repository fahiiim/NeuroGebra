"""Tests for activation function repository."""

import pytest
import numpy as np
from neurogebra.repository.activations import get_activations


class TestActivationRepository:
    """Test activation function repository."""

    def test_get_activations_returns_dict(self):
        """Test that get_activations returns a dictionary."""
        acts = get_activations()
        assert isinstance(acts, dict)
        assert len(acts) > 0

    def test_all_are_expressions(self):
        """Test all values are Expression instances."""
        from neurogebra.core.expression import Expression

        acts = get_activations()
        for name, expr in acts.items():
            assert isinstance(expr, Expression), f"{name} is not an Expression"

    def test_all_have_metadata(self):
        """Test all activations have metadata."""
        acts = get_activations()
        for name, expr in acts.items():
            assert "category" in expr.metadata, f"{name} missing category"
            assert expr.metadata["category"] == "activation"
            assert "description" in expr.metadata, f"{name} missing description"

    def test_relu(self):
        """Test ReLU activation."""
        acts = get_activations()
        relu = acts["relu"]
        assert relu.eval(x=5) == 5
        assert relu.eval(x=-5) == 0
        assert relu.eval(x=0) == 0

    def test_sigmoid(self):
        """Test sigmoid activation."""
        acts = get_activations()
        sigmoid = acts["sigmoid"]
        result = sigmoid.eval(x=0)
        assert abs(result - 0.5) < 1e-10

    def test_sigmoid_bounds(self):
        """Test sigmoid is bounded in [0, 1]."""
        acts = get_activations()
        sigmoid = acts["sigmoid"]
        assert 0 < sigmoid.eval(x=-100) <= 1
        assert 0 <= sigmoid.eval(x=100) <= 1

    def test_tanh(self):
        """Test tanh activation."""
        acts = get_activations()
        tanh = acts["tanh"]
        assert abs(tanh.eval(x=0)) < 1e-10

    def test_tanh_bounds(self):
        """Test tanh is bounded in [-1, 1]."""
        acts = get_activations()
        tanh = acts["tanh"]
        assert -1 <= tanh.eval(x=-100) <= 1
        assert -1 <= tanh.eval(x=100) <= 1

    def test_leaky_relu(self):
        """Test leaky ReLU with default alpha."""
        acts = get_activations()
        leaky = acts["leaky_relu"]
        assert leaky.eval(x=5) == 5
        result = leaky.eval(x=-10)
        assert result == -0.1  # alpha=0.01, so 0.01 * -10

    def test_swish(self):
        """Test swish activation."""
        acts = get_activations()
        swish = acts["swish"]
        result = swish.eval(x=0)
        assert abs(result) < 1e-10  # swish(0) = 0 * sigmoid(0) = 0

    def test_softplus(self):
        """Test softplus activation."""
        acts = get_activations()
        softplus = acts["softplus"]
        result = softplus.eval(x=0)
        assert abs(result - np.log(2)) < 1e-10

    def test_linear_identity(self):
        """Test linear (identity) activation."""
        acts = get_activations()
        linear = acts["linear"]
        assert linear.eval(x=42) == 42
