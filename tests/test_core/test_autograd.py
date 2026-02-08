"""Tests for autograd module."""

import pytest
import numpy as np
from neurogebra.core.autograd import Value, Tensor


class TestValueBasic:
    """Test basic Value operations."""

    def test_creation(self):
        """Test Value creation."""
        v = Value(3.0)
        assert v.data == 3.0
        assert v.grad == 0.0

    def test_repr(self):
        """Test Value string representation."""
        v = Value(2.0)
        assert "2.0" in repr(v)

    def test_hash(self):
        """Test Value is hashable."""
        v = Value(1.0)
        assert hash(v) is not None


class TestValueArithmetic:
    """Test Value arithmetic operations."""

    def test_addition(self):
        """Test value addition."""
        a = Value(2.0)
        b = Value(3.0)
        c = a + b
        assert c.data == 5.0

    def test_scalar_addition(self):
        """Test adding scalar to value."""
        a = Value(2.0)
        c = a + 3
        assert c.data == 5.0

    def test_right_addition(self):
        """Test right scalar addition."""
        a = Value(2.0)
        c = 3 + a
        assert c.data == 5.0

    def test_multiplication(self):
        """Test value multiplication."""
        a = Value(2.0)
        b = Value(3.0)
        c = a * b
        assert c.data == 6.0

    def test_scalar_multiplication(self):
        """Test scalar multiplication."""
        a = Value(2.0)
        c = a * 3
        assert c.data == 6.0

    def test_subtraction(self):
        """Test value subtraction."""
        a = Value(5.0)
        b = Value(3.0)
        c = a - b
        assert c.data == 2.0

    def test_division(self):
        """Test value division."""
        a = Value(6.0)
        b = Value(3.0)
        c = a / b
        assert abs(c.data - 2.0) < 1e-10

    def test_power(self):
        """Test value power."""
        a = Value(3.0)
        c = a ** 2
        assert c.data == 9.0

    def test_negation(self):
        """Test value negation."""
        a = Value(3.0)
        c = -a
        assert c.data == -3.0


class TestValueActivations:
    """Test Value activation functions."""

    def test_relu_positive(self):
        """Test ReLU with positive input."""
        a = Value(3.0)
        c = a.relu()
        assert c.data == 3.0

    def test_relu_negative(self):
        """Test ReLU with negative input."""
        a = Value(-3.0)
        c = a.relu()
        assert c.data == 0.0

    def test_sigmoid(self):
        """Test sigmoid activation."""
        a = Value(0.0)
        c = a.sigmoid()
        assert abs(c.data - 0.5) < 1e-10

    def test_tanh(self):
        """Test tanh activation."""
        a = Value(0.0)
        c = a.tanh()
        assert abs(c.data) < 1e-10

    def test_exp(self):
        """Test exponential."""
        a = Value(1.0)
        c = a.exp()
        assert abs(c.data - np.e) < 1e-10

    def test_log(self):
        """Test natural log."""
        a = Value(np.e)
        c = a.log()
        assert abs(c.data - 1.0) < 1e-10


class TestValueBackprop:
    """Test Value backpropagation."""

    def test_simple_gradient(self):
        """Test gradient of simple expression."""
        a = Value(2.0)
        b = Value(3.0)
        c = a * b  # c = 6
        c.backward()
        assert a.grad == 3.0  # dc/da = b
        assert b.grad == 2.0  # dc/db = a

    def test_addition_gradient(self):
        """Test gradient through addition."""
        a = Value(2.0)
        b = Value(3.0)
        c = a + b
        c.backward()
        assert a.grad == 1.0
        assert b.grad == 1.0

    def test_power_gradient(self):
        """Test gradient through power."""
        a = Value(3.0)
        c = a ** 2
        c.backward()
        assert abs(a.grad - 6.0) < 1e-10  # d(x^2)/dx = 2x = 6

    def test_complex_expression(self):
        """Test gradient through complex expression."""
        x = Value(2.0)
        y = x ** 2 + 3 * x + 1  # y = x^2 + 3x + 1
        y.backward()
        # dy/dx = 2x + 3 = 7
        assert abs(x.grad - 7.0) < 1e-10

    def test_relu_gradient_positive(self):
        """Test ReLU gradient for positive input."""
        a = Value(2.0)
        c = a.relu()
        c.backward()
        assert a.grad == 1.0

    def test_relu_gradient_negative(self):
        """Test ReLU gradient for negative input."""
        a = Value(-2.0)
        c = a.relu()
        c.backward()
        assert a.grad == 0.0

    def test_sigmoid_gradient(self):
        """Test sigmoid gradient."""
        a = Value(0.0)
        c = a.sigmoid()
        c.backward()
        # sigmoid'(0) = 0.5 * 0.5 = 0.25
        assert abs(a.grad - 0.25) < 1e-10

    def test_zero_grad(self):
        """Test zeroing gradients."""
        a = Value(2.0)
        c = a * 3
        c.backward()
        assert a.grad == 3.0
        a.zero_grad()
        assert a.grad == 0.0


class TestTensor:
    """Test Tensor class."""

    def test_creation(self):
        """Test tensor creation."""
        t = Tensor([1.0, 2.0, 3.0])
        assert t.shape == (3,)

    def test_creation_2d(self):
        """Test 2D tensor creation."""
        t = Tensor([[1, 2], [3, 4]])
        assert t.shape == (2, 2)

    def test_requires_grad(self):
        """Test gradient tracking flag."""
        t = Tensor([1.0, 2.0], requires_grad=True)
        assert t.requires_grad is True
        assert t.grad is not None
        np.testing.assert_array_equal(t.grad, [0.0, 0.0])

    def test_sum(self):
        """Test tensor sum."""
        t = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        s = t.sum()
        assert s.data == 6.0

    def test_mean(self):
        """Test tensor mean."""
        t = Tensor([2.0, 4.0, 6.0], requires_grad=True)
        m = t.mean()
        assert m.data == 4.0

    def test_addition(self):
        """Test tensor addition."""
        a = Tensor([1.0, 2.0])
        b = Tensor([3.0, 4.0])
        c = a + b
        np.testing.assert_array_equal(c.data, [4.0, 6.0])

    def test_multiplication(self):
        """Test tensor element-wise multiplication."""
        a = Tensor([2.0, 3.0])
        b = Tensor([4.0, 5.0])
        c = a * b
        np.testing.assert_array_equal(c.data, [8.0, 15.0])

    def test_power(self):
        """Test tensor power."""
        t = Tensor([2.0, 3.0])
        c = t ** 2
        np.testing.assert_array_equal(c.data, [4.0, 9.0])

    def test_repr(self):
        """Test tensor repr."""
        t = Tensor([1.0, 2.0])
        assert "shape" in repr(t)
