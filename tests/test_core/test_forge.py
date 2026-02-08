"""Tests for MathForge."""

import pytest
from neurogebra import MathForge
from neurogebra.core.expression import Expression


class TestForgeInitialization:
    """Test MathForge initialization."""

    def test_creates_successfully(self):
        """Test MathForge creates successfully."""
        forge = MathForge()
        assert forge is not None

    def test_has_expressions(self):
        """Test forge loads expressions on init."""
        forge = MathForge()
        all_exprs = forge.list_all()
        assert len(all_exprs) > 0

    def test_has_activations(self):
        """Test forge has activation functions."""
        forge = MathForge()
        activations = forge.list_all(category="activation")
        assert "relu" in activations
        assert "sigmoid" in activations
        assert "tanh" in activations

    def test_has_losses(self):
        """Test forge has loss functions."""
        forge = MathForge()
        losses = forge.list_all(category="loss")
        assert "mse" in losses
        assert "mae" in losses


class TestForgeGet:
    """Test getting expressions from forge."""

    def test_get_relu(self):
        """Test getting ReLU activation."""
        forge = MathForge()
        relu = forge.get("relu")
        assert relu.name == "relu"
        assert relu.eval(x=-5) == 0
        assert relu.eval(x=5) == 5

    def test_get_sigmoid(self):
        """Test getting sigmoid activation."""
        forge = MathForge()
        sigmoid = forge.get("sigmoid")
        result = sigmoid.eval(x=0)
        assert abs(result - 0.5) < 1e-10

    def test_get_with_params(self):
        """Test getting expression with custom parameters."""
        forge = MathForge()
        leaky = forge.get("leaky_relu", params={"alpha": 0.1})
        result = leaky.eval(x=-10)
        assert result == -1.0  # 0.1 * -10

    def test_get_nonexistent_raises(self):
        """Test that getting nonexistent expression raises KeyError."""
        forge = MathForge()
        with pytest.raises(KeyError):
            forge.get("nonexistent_expression")

    def test_get_trainable(self):
        """Test getting expression with trainable mode."""
        forge = MathForge()
        leaky = forge.get("leaky_relu", trainable=True)
        assert len(leaky.trainable_params) > 0


class TestForgeSearch:
    """Test search functionality."""

    def test_search_relu(self):
        """Test searching for relu."""
        forge = MathForge()
        results = forge.search("relu")
        assert "relu" in results
        assert "leaky_relu" in results

    def test_search_by_description(self):
        """Test searching by description."""
        forge = MathForge()
        results = forge.search("classification")
        assert len(results) > 0

    def test_search_empty_query(self):
        """Test searching with empty query returns nothing."""
        forge = MathForge()
        results = forge.search("")
        # Empty string is in every string, should return all
        assert len(results) > 0

    def test_search_no_results(self):
        """Test searching with no matches."""
        forge = MathForge()
        results = forge.search("zzzznonexistent12345")
        assert len(results) == 0


class TestForgeListAll:
    """Test listing functionality."""

    def test_list_all(self):
        """Test listing all expressions."""
        forge = MathForge()
        all_exprs = forge.list_all()
        assert len(all_exprs) > 0
        assert "relu" in all_exprs

    def test_list_by_category(self):
        """Test listing by category."""
        forge = MathForge()
        activations = forge.list_all(category="activation")
        losses = forge.list_all(category="loss")
        assert len(activations) > 0
        assert len(losses) > 0
        assert "relu" in activations
        assert "mse" in losses

    def test_list_unknown_category(self):
        """Test listing unknown category returns empty."""
        forge = MathForge()
        result = forge.list_all(category="nonexistent_category")
        assert result == []


class TestForgeCompose:
    """Test expression composition."""

    def test_compose_simple(self):
        """Test simple composition string."""
        forge = MathForge()
        composed = forge.compose("mse + mae")
        assert composed is not None

    def test_compose_with_scalar(self):
        """Test composition with scalar weight."""
        forge = MathForge()
        composed = forge.compose("mse + 0.1*mae")
        assert composed is not None


class TestForgeRegister:
    """Test custom expression registration."""

    def test_register_custom(self):
        """Test registering a custom expression."""
        forge = MathForge()
        custom = Expression("my_func", "x**3 + x")
        forge.register("my_func", custom)
        assert "my_func" in forge.list_all()

    def test_use_registered(self):
        """Test using a registered expression."""
        forge = MathForge()
        custom = Expression("my_func", "x**3")
        forge.register("my_func", custom)
        retrieved = forge.get("my_func")
        assert retrieved.eval(x=2) == 8.0


class TestForgeExplain:
    """Test explanation feature."""

    def test_explain(self):
        """Test getting explanation."""
        forge = MathForge()
        explanation = forge.explain("relu")
        assert "relu" in explanation.lower() or "ReLU" in explanation

    def test_compare(self):
        """Test comparing expressions."""
        forge = MathForge()
        comparison = forge.compare(["relu", "sigmoid", "tanh"])
        assert "relu" in comparison
        assert "sigmoid" in comparison
