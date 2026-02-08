"""Integration tests for Neurogebra."""

import pytest
import numpy as np


class TestEndToEndWorkflow:
    """Test complete workflows from user perspective."""

    def test_basic_workflow(self):
        """Test the most basic usage workflow."""
        from neurogebra import MathForge

        forge = MathForge()
        relu = forge.get("relu")
        result = relu.eval(x=5)
        assert result == 5

    def test_activation_comparison_workflow(self):
        """Test comparing multiple activations."""
        from neurogebra import MathForge

        forge = MathForge()

        # Get several activations
        activations = ["relu", "sigmoid", "tanh", "swish"]
        x_val = 1.0

        results = {}
        for name in activations:
            expr = forge.get(name)
            results[name] = expr.eval(x=x_val)

        # All should return finite values
        for name, val in results.items():
            assert np.isfinite(val), f"{name} returned non-finite value"

    def test_gradient_workflow(self):
        """Test gradient computation workflow."""
        from neurogebra import MathForge

        forge = MathForge()

        # Get expression and compute gradient
        sigmoid = forge.get("sigmoid")
        grad = sigmoid.gradient("x")

        # Evaluate gradient
        result = grad.eval(x=0)

        # Sigmoid'(0) = 0.25
        assert abs(result - 0.25) < 1e-10

    def test_training_workflow(self):
        """Test expression training workflow."""
        from neurogebra import MathForge, Expression
        from neurogebra.core.trainer import Trainer

        np.random.seed(42)

        # Create trainable expression
        expr = Expression(
            "linear_fit",
            "a*x + b",
            params={"a": 0.0, "b": 0.0},
            trainable_params=["a", "b"],
        )

        # Train on data
        X = np.linspace(0, 5, 20)
        y = 2 * X + 3 + np.random.normal(0, 0.1, 20)

        trainer = Trainer(expr, learning_rate=0.005)
        history = trainer.fit(X, y, epochs=100, verbose=False)

        # Loss should decrease
        assert history["loss"][-1] < history["loss"][0]

    def test_search_and_use_workflow(self):
        """Test searching and using expressions."""
        from neurogebra import MathForge

        forge = MathForge()

        # Search for activation functions
        results = forge.search("activation")
        assert len(results) > 0

        # Use first result
        expr = forge.get(results[0])
        result = expr.eval(x=1.0)
        assert np.isfinite(result)

    def test_custom_expression_workflow(self):
        """Test creating and using custom expressions."""
        from neurogebra import MathForge, Expression

        forge = MathForge()

        # Create custom activation
        custom = Expression(
            "my_activation",
            "x * tanh(x)",
            metadata={
                "category": "activation",
                "description": "Custom x*tanh(x) activation",
            },
        )

        # Register it
        forge.register("my_activation", custom)

        # Retrieve and use it
        retrieved = forge.get("my_activation")
        result = retrieved.eval(x=1.0)
        expected = 1.0 * np.tanh(1.0)
        assert abs(result - expected) < 1e-10

    def test_expression_explain_workflow(self):
        """Test explanation workflow."""
        from neurogebra import MathForge

        forge = MathForge()

        # Get and explain
        explanation = forge.explain("relu", level="intermediate")
        assert isinstance(explanation, str)
        assert len(explanation) > 0

    def test_algebra_expressions(self):
        """Test algebra expressions from repository."""
        from neurogebra import MathForge

        forge = MathForge()

        # Quadratic
        quad = forge.get("quadratic", params={"a": 1, "b": 0, "c": 0})
        assert quad.eval(x=3) == 9.0

    def test_autograd_integration(self):
        """Test autograd with expression-like operations."""
        from neurogebra.core.autograd import Value

        # Build simple neural network computation
        x = Value(2.0)
        w = Value(3.0)

        # Forward: y = w*x + bias
        y = w * x + Value(1.0)
        assert y.data == 7.0

        # Backward
        y.backward()
        assert w.grad == 2.0  # dy/dw = x
        assert x.grad == 3.0  # dy/dx = w

    def test_expression_composition_chain(self):
        """Test chaining multiple compositions."""
        from neurogebra import Expression

        f = Expression("f", "x**2")
        g = Expression("g", "x + 1")

        # f(g(x)) = (x+1)^2
        h = f.compose(g)
        result = h.eval(x=2)
        assert result == 9.0  # (2+1)^2

    def test_list_and_compare(self):
        """Test listing and comparing workflow."""
        from neurogebra import MathForge

        forge = MathForge()

        # List categories
        activations = forge.list_all(category="activation")
        losses = forge.list_all(category="loss")

        assert len(activations) >= 5
        assert len(losses) >= 3

        # Compare
        comparison = forge.compare(["relu", "sigmoid", "tanh"])
        assert "relu" in comparison
