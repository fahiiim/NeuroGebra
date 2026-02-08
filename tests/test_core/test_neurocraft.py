"""Tests for NeuroCraft."""

import pytest
from neurogebra.core.neurocraft import NeuroCraft
from neurogebra.core.expression import Expression


class TestNeuroCraftInitialization:
    """Test NeuroCraft initialization."""

    def test_creates_successfully(self):
        """Test NeuroCraft creates successfully."""
        craft = NeuroCraft(educational_mode=False)
        assert craft is not None

    def test_educational_mode_off(self):
        """Test NeuroCraft with educational mode off."""
        craft = NeuroCraft(educational_mode=False)
        assert craft.educational_mode is False

    def test_educational_mode_on(self, capsys):
        """Test NeuroCraft with educational mode shows welcome."""
        craft = NeuroCraft(educational_mode=True)
        assert craft.educational_mode is True
        captured = capsys.readouterr()
        assert "Welcome" in captured.out

    def test_has_repository(self):
        """Test NeuroCraft loads repository on init."""
        craft = NeuroCraft(educational_mode=False)
        assert len(craft._repository) > 0


class TestNeuroCraftGet:
    """Test getting expressions from NeuroCraft."""

    def test_get_relu(self):
        """Test getting ReLU activation."""
        craft = NeuroCraft(educational_mode=False)
        relu = craft.get("relu")
        assert relu.name == "relu"
        assert relu.eval(x=-5) == 0
        assert relu.eval(x=5) == 5

    def test_get_sigmoid(self):
        """Test getting sigmoid activation."""
        craft = NeuroCraft(educational_mode=False)
        sigmoid = craft.get("sigmoid")
        result = sigmoid.eval(x=0)
        assert abs(result - 0.5) < 1e-10

    def test_get_mse(self):
        """Test getting MSE loss."""
        craft = NeuroCraft(educational_mode=False)
        mse = craft.get("mse")
        assert mse.name == "mse"

    def test_get_nonexistent_raises(self):
        """Test getting a nonexistent expression raises KeyError."""
        craft = NeuroCraft(educational_mode=False)
        with pytest.raises(KeyError, match="not found"):
            craft.get("nonexistent_function")

    def test_get_with_suggestions(self):
        """Test that KeyError includes suggestions for similar names."""
        craft = NeuroCraft(educational_mode=False)
        with pytest.raises(KeyError, match="Did you mean"):
            craft.get("reluu")

    def test_get_returns_clone(self):
        """Test that get returns a cloned expression, not original."""
        craft = NeuroCraft(educational_mode=False)
        relu1 = craft.get("relu")
        relu2 = craft.get("relu")
        assert relu1 is not relu2

    def test_get_with_params(self):
        """Test getting expression with custom parameters."""
        craft = NeuroCraft(educational_mode=False)
        expr = craft.get("leaky_relu", params={"alpha": 0.1})
        assert expr.params.get("alpha") == 0.1

    def test_get_with_explain(self, capsys):
        """Test getting with explain=True prints explanation."""
        craft = NeuroCraft(educational_mode=True)
        _ = capsys.readouterr()  # clear welcome message
        craft.get("relu", explain=True)
        captured = capsys.readouterr()
        assert len(captured.out) > 0


class TestNeuroCraftSearch:
    """Test NeuroCraft search functionality."""

    def test_search_activation(self):
        """Test searching for activations."""
        craft = NeuroCraft(educational_mode=False)
        results = craft.search("relu", show_details=False)
        assert "relu" in results

    def test_search_by_category(self):
        """Test searching with category filter."""
        craft = NeuroCraft(educational_mode=False)
        results = craft.search("", category="activation", show_details=False)
        assert len(results) > 0
        # Verify all results are activations
        for name in results:
            expr = craft._repository[name]
            assert expr.metadata.get("category") == "activation"

    def test_search_no_results(self):
        """Test search with no matches returns empty list."""
        craft = NeuroCraft(educational_mode=False)
        results = craft.search("zzzznonexistent", show_details=False)
        assert results == []


class TestNeuroCraftListAll:
    """Test listing expressions."""

    def test_list_all(self):
        """Test listing all expressions."""
        craft = NeuroCraft(educational_mode=False)
        all_exprs = craft.list_all()
        assert len(all_exprs) > 0
        assert "relu" in all_exprs
        assert "mse" in all_exprs

    def test_list_by_category(self):
        """Test listing by category."""
        craft = NeuroCraft(educational_mode=False)
        activations = craft.list_all(category="activation")
        assert "relu" in activations
        assert "sigmoid" in activations
        # losses should not appear
        assert "mse" not in activations

    def test_list_losses(self):
        """Test listing loss functions."""
        craft = NeuroCraft(educational_mode=False)
        losses = craft.list_all(category="loss")
        assert "mse" in losses
        assert "mae" in losses


class TestNeuroCraftRegister:
    """Test registering custom expressions."""

    def test_register_custom(self):
        """Test registering a custom expression."""
        craft = NeuroCraft(educational_mode=False)
        expr = Expression("custom", "x**3")
        craft.register("custom_cube", expr)
        assert "custom_cube" in craft.list_all()
        result = craft.get("custom_cube")
        assert result.eval(x=2) == 8.0


class TestNeuroCraftCompose:
    """Test expression composition."""

    def test_compose_simple(self):
        """Test simple expression composition."""
        craft = NeuroCraft(educational_mode=False)
        composed = craft.compose("mse + 0.1*mae")
        assert composed is not None

    def test_compose_single(self):
        """Test composing a single expression."""
        craft = NeuroCraft(educational_mode=False)
        composed = craft.compose("mse")
        assert composed is not None


class TestNeuroCraftQuickMethods:
    """Test quick access methods."""

    def test_quick_activation(self):
        """Test quick_activation returns an expression."""
        craft = NeuroCraft(educational_mode=False)
        act = craft.quick_activation("relu")
        assert act.name == "relu"

    def test_quick_loss(self):
        """Test quick_loss returns an expression."""
        craft = NeuroCraft(educational_mode=False)
        loss = craft.quick_loss("mse")
        assert loss.name == "mse"

    def test_quick_activation_default(self):
        """Test quick_activation with default is relu."""
        craft = NeuroCraft(educational_mode=False)
        act = craft.quick_activation()
        assert act.name == "relu"


class TestNeuroCraftFindSimilar:
    """Test the similarity search."""

    def test_find_similar_relu(self):
        """Test finding names similar to relu."""
        craft = NeuroCraft(educational_mode=False)
        similar = craft._find_similar("relu")
        assert isinstance(similar, list)
        assert "relu" in similar

    def test_find_similar_empty(self):
        """Test with completely unrelated string."""
        craft = NeuroCraft(educational_mode=False)
        similar = craft._find_similar("zzz")
        assert isinstance(similar, list)


class TestNeuroCraftTutorial:
    """Test tutorial integration."""

    def test_tutorial_menu(self, capsys):
        """Test showing tutorial menu."""
        craft = NeuroCraft(educational_mode=False)
        craft.tutorial()
        captured = capsys.readouterr()
        assert "tutorials" in captured.out.lower() or "Tutorial" in captured.out

    def test_tutorial_specific(self, capsys):
        """Test starting a specific tutorial."""
        craft = NeuroCraft(educational_mode=False)
        craft.tutorial("basics")
        captured = capsys.readouterr()
        assert "Neural Network" in captured.out or "STEP" in captured.out
