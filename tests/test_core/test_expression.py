"""Tests for Expression class."""

import pytest
import numpy as np
from neurogebra.core.expression import Expression


class TestExpressionCreation:
    """Test expression creation and initialization."""

    def test_basic_creation(self):
        """Test basic expression creation from string."""
        expr = Expression("test", "x**2 + 2*x + 1")
        assert expr.name == "test"
        assert str(expr.symbolic_expr) == "x**2 + 2*x + 1"

    def test_creation_with_params(self):
        """Test expression creation with parameters."""
        expr = Expression(
            "quadratic",
            "a*x**2 + b*x + c",
            params={"a": 1, "b": 2, "c": 3},
        )
        assert expr.params == {"a": 1, "b": 2, "c": 3}

    def test_creation_with_trainable(self):
        """Test expression creation with trainable parameters."""
        expr = Expression(
            "linear",
            "m*x + b",
            params={"m": 1, "b": 0},
            trainable_params=["m", "b"],
        )
        assert expr.trainable_params == ["m", "b"]

    def test_creation_with_metadata(self):
        """Test expression creation with metadata."""
        expr = Expression(
            "test",
            "x**2",
            metadata={"description": "Square function", "category": "algebra"},
        )
        assert expr.metadata["description"] == "Square function"
        assert expr.metadata["category"] == "algebra"

    def test_variables_extraction(self):
        """Test that free variables are extracted correctly."""
        expr = Expression("multi", "x**2 + y**2")
        var_names = [str(v) for v in expr.variables]
        assert "x" in var_names
        assert "y" in var_names

    def test_constant_expression(self):
        """Test constant expression (no variables)."""
        expr = Expression("const", "42")
        assert expr.variables == []
        result = expr.eval()
        assert result == 42.0


class TestExpressionEval:
    """Test expression evaluation."""

    def test_simple_eval(self):
        """Test simple numerical evaluation."""
        expr = Expression("square", "x**2")
        assert expr.eval(x=3) == 9.0

    def test_eval_with_params(self):
        """Test evaluation with parameter substitution."""
        expr = Expression(
            "quadratic",
            "a*x**2 + b*x + c",
            params={"a": 1, "b": 2, "c": 1},
        )
        result = expr.eval(x=2)
        expected = 1 * 4 + 2 * 2 + 1  # 9
        assert result == expected

    def test_eval_callable(self):
        """Test calling expression as function."""
        expr = Expression("square", "x**2")
        assert expr(x=4) == 16.0

    def test_eval_with_numpy_array(self):
        """Test evaluation with numpy arrays."""
        expr = Expression("relu", "Max(0, x)")
        result = expr.eval(x=np.array([-1, 0, 1, 2]))
        expected = np.array([0, 0, 1, 2])
        np.testing.assert_array_equal(result, expected)

    def test_eval_negative_values(self):
        """Test evaluation with negative inputs."""
        expr = Expression("abs", "Abs(x)")
        assert expr.eval(x=-5) == 5.0


class TestExpressionGradient:
    """Test gradient computation."""

    def test_basic_gradient(self):
        """Test gradient of x^2."""
        expr = Expression("square", "x**2")
        grad = expr.gradient("x")
        assert str(grad.symbolic_expr) == "2*x"

    def test_gradient_evaluation(self):
        """Test gradient numerical evaluation."""
        expr = Expression("square", "x**2")
        grad = expr.gradient("x")
        assert grad.eval(x=3) == 6.0

    def test_gradient_constant(self):
        """Test gradient of constant is zero."""
        expr = Expression("const", "5")
        grad = expr.gradient("x")
        assert grad.eval() == 0.0

    def test_gradient_linear(self):
        """Test gradient of linear function."""
        expr = Expression("linear", "3*x + 2")
        grad = expr.gradient("x")
        # d/dx(3x + 2) = 3
        assert grad.eval(x=0) == 3.0

    def test_gradient_name(self):
        """Test gradient name formatting."""
        expr = Expression("f", "x**2")
        grad = expr.gradient("x")
        assert "d(" in grad.name


class TestExpressionComposition:
    """Test expression composition."""

    def test_basic_composition(self):
        """Test composing two expressions."""
        f = Expression("f", "x**2")
        g = Expression("g", "x + 1")
        composed = f.compose(g)
        # f(g(x)) = (x+1)^2
        result = composed.eval(x=2)
        assert result == 9.0  # (2+1)^2

    def test_composition_name(self):
        """Test composed expression name."""
        f = Expression("f", "x**2")
        g = Expression("g", "x + 1")
        composed = f.compose(g)
        assert "f" in composed.name
        assert "g" in composed.name


class TestExpressionArithmetic:
    """Test arithmetic operations."""

    def test_addition(self):
        """Test expression addition."""
        e1 = Expression("f", "x**2")
        e2 = Expression("g", "2*x")
        result = e1 + e2
        assert result.eval(x=3) == 15.0  # 9 + 6

    def test_scalar_addition(self):
        """Test adding scalar to expression."""
        e = Expression("f", "x**2")
        result = e + 5
        assert result.eval(x=2) == 9.0  # 4 + 5

    def test_multiplication(self):
        """Test expression multiplication."""
        e1 = Expression("f", "x")
        e2 = Expression("g", "x")
        result = e1 * e2
        assert result.eval(x=3) == 9.0  # 3 * 3

    def test_scalar_multiplication(self):
        """Test multiplying expression by scalar."""
        e = Expression("f", "x")
        result = e * 3
        assert result.eval(x=4) == 12.0

    def test_right_scalar_multiplication(self):
        """Test scalar * expression."""
        e = Expression("f", "x")
        result = 3 * e
        assert result.eval(x=4) == 12.0

    def test_subtraction(self):
        """Test expression subtraction."""
        e1 = Expression("f", "x**2")
        e2 = Expression("g", "x")
        result = e1 - e2
        assert result.eval(x=3) == 6.0  # 9 - 3

    def test_negation(self):
        """Test expression negation."""
        e = Expression("f", "x")
        result = -e
        assert result.eval(x=5) == -5.0

    def test_power(self):
        """Test expression power."""
        e = Expression("f", "x")
        result = e ** 3
        assert result.eval(x=2) == 8.0


class TestExpressionProperties:
    """Test expression properties and utilities."""

    def test_formula_latex(self):
        """Test LaTeX formula generation."""
        expr = Expression("test", "x**2 + 1")
        formula = expr.formula
        assert isinstance(formula, str)
        assert len(formula) > 0

    def test_repr(self):
        """Test string representation."""
        expr = Expression("test", "x**2")
        repr_str = repr(expr)
        assert "test" in repr_str
        assert "Expression" in repr_str

    def test_str(self):
        """Test str conversion."""
        expr = Expression("test", "x**2")
        assert "x**2" in str(expr)

    def test_simplify(self):
        """Test expression simplification."""
        expr = Expression("test", "x**2 + 2*x + 1")
        simplified = expr.simplify()
        # (x+1)^2 is the simplified form
        assert simplified.symbolic_expr is not None

    def test_expand(self):
        """Test expression expansion."""
        expr = Expression("test", "(x + 1)**2")
        expanded = expr.expand()
        # Should expand to x^2 + 2x + 1
        assert expanded.eval(x=2) == 9.0

    def test_integrate(self):
        """Test symbolic integration."""
        expr = Expression("test", "2*x")
        integral = expr.integrate("x")
        # integral of 2x = x^2
        assert integral.eval(x=3) == 9.0

    def test_explain(self):
        """Test explain functionality."""
        expr = Expression(
            "test",
            "x**2",
            metadata={"description": "Square function"},
        )
        explanation = expr.explain()
        assert "test" in explanation
        assert "Square function" in explanation

    def test_explain_advanced(self):
        """Test advanced explanation."""
        expr = Expression("test", "x**2")
        explanation = expr.explain(level="advanced")
        assert "2*x" in explanation  # gradient should be shown
