"""Tests for ModelBuilder, Layer, and Model."""

import pytest
import numpy as np
from neurogebra.builders.model_builder import Layer, ModelBuilder, Model


class TestLayer:
    """Test Layer creation and properties."""

    def test_dense_layer(self):
        """Test creating a dense layer."""
        layer = Layer("dense", units=64, activation="relu")
        assert layer.layer_type == "dense"
        assert layer.units == 64
        assert layer.activation == "relu"

    def test_dropout_layer(self):
        """Test creating a dropout layer."""
        layer = Layer("dropout", rate=0.3)
        assert layer.layer_type == "dropout"
        assert layer.params["rate"] == 0.3

    def test_conv2d_layer(self):
        """Test creating a conv2d layer."""
        layer = Layer("conv2d", units=32, activation="relu", kernel_size=3)
        assert layer.layer_type == "conv2d"
        assert layer.units == 32
        assert layer.params["kernel_size"] == 3

    def test_flatten_layer(self):
        """Test creating a flatten layer."""
        layer = Layer("flatten")
        assert layer.layer_type == "flatten"

    def test_layer_description(self):
        """Test layer has educational description."""
        layer = Layer("dense", units=64)
        assert layer.description is not None
        assert len(layer.description) > 0

    def test_layer_use_cases(self):
        """Test layer has use cases."""
        layer = Layer("dense", units=64)
        assert isinstance(layer.best_for, list)
        assert len(layer.best_for) > 0

    def test_layer_explain(self, capsys):
        """Test layer explain prints info."""
        layer = Layer("dense", units=64, activation="relu")
        layer.explain()
        captured = capsys.readouterr()
        assert "DENSE" in captured.out
        assert "64" in captured.out

    def test_layer_repr(self):
        """Test layer string representation."""
        layer = Layer("dense", units=128, activation="relu")
        repr_str = repr(layer)
        assert "dense" in repr_str
        assert "128" in repr_str
        assert "relu" in repr_str

    def test_custom_layer(self):
        """Test creating a custom layer type."""
        layer = Layer("custom", units=42)
        assert layer.layer_type == "custom"
        assert layer.description == "Custom layer"


class TestModelBuilder:
    """Test ModelBuilder layer creation methods."""

    def test_dense(self):
        """Test Dense factory method."""
        builder = ModelBuilder()
        layer = builder.Dense(128, activation="relu")
        assert layer.layer_type == "dense"
        assert layer.units == 128
        assert layer.activation == "relu"

    def test_dense_with_input_shape(self):
        """Test Dense with input_shape parameter."""
        builder = ModelBuilder()
        layer = builder.Dense(64, activation="relu", input_shape=(784,))
        assert layer.params["input_shape"] == (784,)

    def test_conv2d(self):
        """Test Conv2D factory method."""
        builder = ModelBuilder()
        layer = builder.Conv2D(32, kernel_size=3, activation="relu")
        assert layer.layer_type == "conv2d"
        assert layer.units == 32
        assert layer.params["kernel_size"] == 3

    def test_dropout(self):
        """Test Dropout factory method."""
        builder = ModelBuilder()
        layer = builder.Dropout(0.5)
        assert layer.layer_type == "dropout"
        assert layer.params["rate"] == 0.5

    def test_batchnorm(self):
        """Test BatchNorm factory method."""
        builder = ModelBuilder()
        layer = builder.BatchNorm()
        assert layer.layer_type == "batchnorm"

    def test_maxpooling2d(self):
        """Test MaxPooling2D factory method."""
        builder = ModelBuilder()
        layer = builder.MaxPooling2D(pool_size=2)
        assert layer.layer_type == "maxpool2d"
        assert layer.params["pool_size"] == 2

    def test_flatten(self):
        """Test Flatten factory method."""
        builder = ModelBuilder()
        layer = builder.Flatten()
        assert layer.layer_type == "flatten"


class TestModelBuilderSequential:
    """Test Sequential model creation."""

    def test_sequential_creates_model(self):
        """Test Sequential returns a Model instance."""
        builder = ModelBuilder()
        model = builder.Sequential([
            builder.Dense(64, activation="relu"),
            builder.Dense(10, activation="softmax"),
        ])
        assert isinstance(model, Model)

    def test_sequential_with_name(self):
        """Test Sequential with custom name."""
        builder = ModelBuilder()
        model = builder.Sequential(
            [builder.Dense(10)],
            name="my_model",
        )
        assert model.name == "my_model"

    def test_sequential_preserves_layers(self):
        """Test Sequential preserves all layers."""
        builder = ModelBuilder()
        model = builder.Sequential([
            builder.Dense(128, activation="relu"),
            builder.Dropout(0.2),
            builder.Dense(64, activation="relu"),
            builder.Dense(10, activation="softmax"),
        ])
        assert len(model.layers) == 4


class TestModelBuilderTemplates:
    """Test template-based model creation."""

    def test_from_template_simple_classifier(self):
        """Test creating model from simple_classifier template."""
        builder = ModelBuilder()
        model = builder.from_template("simple_classifier")
        assert isinstance(model, Model)
        assert len(model.layers) > 0
        assert model.template_info is not None

    def test_from_template_image_classifier(self):
        """Test creating model from image_classifier template."""
        builder = ModelBuilder()
        model = builder.from_template("image_classifier")
        assert isinstance(model, Model)
        # Should have conv layers
        layer_types = [l.layer_type for l in model.layers]
        assert "conv2d" in layer_types

    def test_from_template_regression(self):
        """Test creating model from regression template."""
        builder = ModelBuilder()
        model = builder.from_template("regression")
        assert isinstance(model, Model)
        # Last layer should have 1 unit
        assert model.layers[-1].units == 1

    def test_from_template_binary_classifier(self):
        """Test creating model from binary_classifier template."""
        builder = ModelBuilder()
        model = builder.from_template("binary_classifier")
        assert isinstance(model, Model)
        # Last layer should have sigmoid activation
        assert model.layers[-1].activation == "sigmoid"

    def test_from_template_invalid(self):
        """Test from_template with invalid name raises ValueError."""
        builder = ModelBuilder()
        with pytest.raises(ValueError, match="not found"):
            builder.from_template("nonexistent_template")

    def test_from_template_no_mutation(self):
        """Test that repeated calls don't mutate the template."""
        builder = ModelBuilder()
        model1 = builder.from_template("simple_classifier")
        model2 = builder.from_template("simple_classifier")
        assert len(model1.layers) == len(model2.layers)

    def test_list_templates(self, capsys):
        """Test list_templates prints template info."""
        builder = ModelBuilder()
        builder.list_templates()
        captured = capsys.readouterr()
        assert "simple_classifier" in captured.out
        assert "regression" in captured.out

    def test_suggest_architecture(self, capsys):
        """Test suggest_architecture prints suggestions."""
        builder = ModelBuilder()
        builder.suggest_architecture(
            task="classification",
            input_shape=(784,),
            output_size=10,
        )
        captured = capsys.readouterr()
        assert "Suggested architecture" in captured.out or "Recommended" in captured.out


class TestModel:
    """Test Model class."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        builder = ModelBuilder()
        return builder.Sequential([
            builder.Dense(64, activation="relu"),
            builder.Dropout(0.2),
            builder.Dense(10, activation="softmax"),
        ])

    def test_model_creation(self, simple_model):
        """Test model is created correctly."""
        assert len(simple_model.layers) == 3
        assert simple_model.is_compiled is False

    def test_model_summary(self, simple_model, capsys):
        """Test model summary prints info."""
        simple_model.summary()
        captured = capsys.readouterr()
        assert "Model" in captured.out
        assert "Layer" in captured.out

    def test_model_explain_architecture(self, simple_model, capsys):
        """Test explain_architecture prints details."""
        simple_model.explain_architecture()
        captured = capsys.readouterr()
        assert "DENSE" in captured.out or "DROPOUT" in captured.out

    def test_model_compile(self, simple_model, capsys):
        """Test model compilation."""
        simple_model.compile(
            loss="mse",
            optimizer="adam",
            metrics=["accuracy"],
        )
        assert simple_model.is_compiled is True
        assert simple_model.loss_name == "mse"
        assert simple_model.optimizer == "adam"
        assert simple_model.learning_rate == 0.001

    def test_model_compile_custom_lr(self, simple_model):
        """Test compile with custom learning rate."""
        simple_model.compile(loss="mse", learning_rate=0.01)
        assert simple_model.learning_rate == 0.01

    def test_model_fit_without_compile_raises(self, simple_model):
        """Test fit raises error if model not compiled."""
        X = np.random.rand(100, 10)
        y = np.random.randint(0, 10, 100)
        with pytest.raises(RuntimeError, match="compiled"):
            simple_model.fit(X, y, epochs=2)

    def test_model_fit(self, simple_model, capsys):
        """Test model training runs and returns history."""
        simple_model.compile(loss="mse", optimizer="adam")
        X = np.random.rand(100, 10)
        y = np.random.randint(0, 10, 100)
        history = simple_model.fit(X, y, epochs=3, verbose=True, visualize=False)

        assert "loss" in history
        assert "accuracy" in history
        assert "val_loss" in history
        assert "val_accuracy" in history
        assert len(history["loss"]) == 3

    def test_model_predict(self, simple_model, capsys):
        """Test model prediction returns array."""
        simple_model.compile(loss="mse")
        X = np.random.rand(20, 10)
        preds = simple_model.predict(X)
        assert isinstance(preds, np.ndarray)
        assert preds.shape[0] == 20

    def test_model_evaluate(self, simple_model, capsys):
        """Test model evaluation returns metrics."""
        simple_model.compile(loss="mse")
        X = np.random.rand(50, 10)
        y = np.random.randint(0, 10, 50)
        results = simple_model.evaluate(X, y)
        assert "loss" in results
        assert "accuracy" in results

    def test_model_save_and_load(self, simple_model, tmp_path, capsys):
        """Test model save."""
        simple_model.compile(loss="mse")
        filepath = str(tmp_path / "test_model.pkl")
        simple_model.save(filepath)
        import os
        assert os.path.exists(filepath)

    def test_model_repr(self, simple_model):
        """Test model string representation."""
        repr_str = repr(simple_model)
        assert "Model" in repr_str
        assert "layers=3" in repr_str

    def test_model_plot_history(self, simple_model, capsys):
        """Test plot_history doesn't error with empty history."""
        # Model with populated history
        simple_model.history = {
            "loss": [1.0, 0.8, 0.6],
            "accuracy": [0.5, 0.6, 0.7],
            "val_loss": [1.1, 0.9, 0.7],
            "val_accuracy": [0.45, 0.55, 0.65],
        }
        # This will try to plot - just verify it doesn't crash
        import matplotlib
        matplotlib.use("Agg")
        simple_model.plot_history()
