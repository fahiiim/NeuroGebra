"""
ModelBuilder: Beginner-friendly neural network construction.

Provides intuitive interfaces for building ML models with educational
features built in.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Union

import numpy as np


class Layer:
    """
    Represents a single layer in a neural network.

    This is an educational abstraction that makes it easy to
    understand what each layer does.

    Examples:
        >>> layer = Layer("dense", units=64, activation="relu")
        >>> layer.explain()
    """

    _DESCRIPTIONS = {
        "dense": (
            "Fully connected layer - every neuron connects to "
            "every neuron in next layer"
        ),
        "conv2d": "Convolutional layer - detects patterns in images",
        "dropout": (
            "Randomly turns off neurons during training to "
            "prevent overfitting"
        ),
        "flatten": "Converts multi-dimensional data to 1D for dense layers",
        "maxpool2d": (
            "Reduces image size by taking maximum values in regions"
        ),
        "batchnorm": (
            "Normalizes activations in a layer to stabilize training"
        ),
    }

    _USE_CASES = {
        "dense": ["Classification", "Regression", "Final output layer"],
        "conv2d": [
            "Image recognition",
            "Object detection",
            "Feature extraction",
        ],
        "dropout": ["Preventing overfitting", "Regularization"],
        "flatten": ["Bridging conv layers to dense layers"],
        "maxpool2d": ["Downsampling images", "Reducing computation"],
        "batchnorm": ["Faster training", "Reducing internal covariate shift"],
    }

    def __init__(
        self,
        layer_type: str,
        units: Optional[int] = None,
        activation: Optional[str] = None,
        **kwargs,
    ):
        """
        Create a layer.

        Args:
            layer_type: Type of layer ('dense', 'conv2d', 'dropout', etc.)
            units: Number of neurons/filters
            activation: Activation function name
            **kwargs: Additional layer-specific parameters
        """
        self.layer_type = layer_type
        self.units = units
        self.activation = activation
        self.params = kwargs

        # Educational metadata
        self.description = self._DESCRIPTIONS.get(
            self.layer_type, "Custom layer"
        )
        self.best_for = self._USE_CASES.get(
            self.layer_type, ["General purpose"]
        )

    def explain(self):
        """Explain what this layer does in plain language."""
        print(f"\n📚 {self.layer_type.upper()} Layer")
        print(f"   {self.description}")
        print(f"\n   Best used for:")
        for use_case in self.best_for:
            print(f"   • {use_case}")

        if self.units:
            print(f"\n   Neurons/Units: {self.units}")
        if self.activation:
            print(f"   Activation: {self.activation}")
        for key, val in self.params.items():
            print(f"   {key}: {val}")

    def __repr__(self):
        parts = [f"{self.layer_type}"]
        if self.units:
            parts.append(f"units={self.units}")
        if self.activation:
            parts.append(f"activation={self.activation}")
        for key, val in self.params.items():
            parts.append(f"{key}={val}")
        return f"Layer({', '.join(parts)})"


class ModelBuilder:
    """
    Build neural networks with an educational, intuitive interface.

    ModelBuilder makes it easy for beginners to:
    - Understand what they're building
    - Get guidance on architecture choices
    - See what each layer does
    - Learn best practices

    Examples:
        >>> from neurogebra.builders import ModelBuilder
        >>> builder = ModelBuilder()
        >>> model = builder.Sequential([
        ...     builder.Dense(128, activation="relu"),
        ...     builder.Dropout(0.2),
        ...     builder.Dense(10, activation="softmax")
        ... ])
        >>> model.summary()
    """

    def __init__(self, craft=None):
        """
        Initialize ModelBuilder.

        Args:
            craft: NeuroCraft instance for expression access.
                   If None, a default one is created internally.
        """
        self.craft = craft
        self._templates = self._load_templates()

    def _get_craft(self):
        """Lazily initialize craft if needed."""
        if self.craft is None:
            from neurogebra.core.neurocraft import NeuroCraft

            self.craft = NeuroCraft(educational_mode=False)
        return self.craft

    def _load_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load pre-built model templates for common tasks."""
        return {
            "simple_classifier": {
                "description": "Basic neural network for classification",
                "layers": [
                    {"type": "dense", "units": 128, "activation": "relu"},
                    {"type": "dropout", "rate": 0.2},
                    {"type": "dense", "units": 64, "activation": "relu"},
                    {"type": "dense", "units": 10, "activation": "softmax"},
                ],
                "good_for": [
                    "MNIST",
                    "CIFAR-10",
                    "Simple tabular data",
                ],
            },
            "image_classifier": {
                "description": "CNN for image classification",
                "layers": [
                    {
                        "type": "conv2d",
                        "filters": 32,
                        "kernel_size": 3,
                        "activation": "relu",
                    },
                    {"type": "maxpool2d", "pool_size": 2},
                    {
                        "type": "conv2d",
                        "filters": 64,
                        "kernel_size": 3,
                        "activation": "relu",
                    },
                    {"type": "maxpool2d", "pool_size": 2},
                    {"type": "flatten"},
                    {"type": "dense", "units": 128, "activation": "relu"},
                    {"type": "dense", "units": 10, "activation": "softmax"},
                ],
                "good_for": [
                    "Image recognition",
                    "Object classification",
                ],
            },
            "regression": {
                "description": "Network for predicting continuous values",
                "layers": [
                    {"type": "dense", "units": 64, "activation": "relu"},
                    {"type": "dense", "units": 32, "activation": "relu"},
                    {"type": "dense", "units": 1, "activation": "linear"},
                ],
                "good_for": [
                    "House price prediction",
                    "Stock prices",
                    "Any regression task",
                ],
            },
            "binary_classifier": {
                "description": "Network for binary yes/no classification",
                "layers": [
                    {"type": "dense", "units": 64, "activation": "relu"},
                    {"type": "dropout", "rate": 0.3},
                    {"type": "dense", "units": 32, "activation": "relu"},
                    {"type": "dense", "units": 1, "activation": "sigmoid"},
                ],
                "good_for": [
                    "Spam detection",
                    "Medical diagnosis",
                    "Sentiment analysis",
                ],
            },
        }

    # ---- Layer creation methods ----

    def Dense(
        self,
        units: int,
        activation: Optional[str] = None,
        input_shape: Optional[tuple] = None,
        **kwargs,
    ) -> Layer:
        """
        Create a fully connected (dense) layer.

        Args:
            units: Number of neurons
            activation: Activation function name
            input_shape: Shape of input (only for first layer)

        Returns:
            Layer instance

        Examples:
            >>> builder = ModelBuilder()
            >>> layer = builder.Dense(128, activation="relu")
            >>> layer.explain()  # Learn what it does
        """
        extra = {**kwargs}
        if input_shape is not None:
            extra["input_shape"] = input_shape
        return Layer("dense", units=units, activation=activation, **extra)

    def Conv2D(
        self,
        filters: int,
        kernel_size: int = 3,
        activation: Optional[str] = None,
        **kwargs,
    ) -> Layer:
        """
        Create a 2D convolutional layer for images.

        Args:
            filters: Number of filters / feature detectors
            kernel_size: Size of the filter window
            activation: Activation function

        Returns:
            Layer instance
        """
        return Layer(
            "conv2d",
            units=filters,
            activation=activation,
            kernel_size=kernel_size,
            **kwargs,
        )

    def Dropout(self, rate: float = 0.2) -> Layer:
        """
        Create a dropout layer for regularization.

        Args:
            rate: Fraction of neurons to drop (0.0 to 1.0)

        Returns:
            Layer instance
        """
        return Layer("dropout", rate=rate)

    def BatchNorm(self) -> Layer:
        """Create a batch normalization layer."""
        return Layer("batchnorm")

    def MaxPooling2D(self, pool_size: int = 2) -> Layer:
        """Create a max pooling layer."""
        return Layer("maxpool2d", pool_size=pool_size)

    def Flatten(self) -> Layer:
        """Create a flatten layer."""
        return Layer("flatten")

    # ---- Model construction ----

    def Sequential(
        self, layers: List[Layer], name: Optional[str] = None
    ) -> "Model":
        """
        Build a sequential model (layers stacked one after another).

        Args:
            layers: List of Layer instances
            name: Optional name for the model

        Returns:
            Model instance ready for compilation and training

        Examples:
            >>> builder = ModelBuilder()
            >>> model = builder.Sequential([
            ...     builder.Dense(128, activation="relu"),
            ...     builder.Dropout(0.2),
            ...     builder.Dense(10, activation="softmax")
            ... ])
            >>> model.summary()
        """
        return Model(layers=layers, name=name, craft=self._get_craft())

    def from_template(
        self,
        template_name: str,
        customize: Optional[Dict] = None,
    ) -> "Model":
        """
        Create a model from a pre-built template.

        Args:
            template_name: Name of template (e.g. 'simple_classifier')
            customize: Optional customizations (not yet implemented)

        Returns:
            Model instance

        Examples:
            >>> builder = ModelBuilder()
            >>> model = builder.from_template("simple_classifier")
            >>> model.explain_architecture()
        """
        if template_name not in self._templates:
            available = ", ".join(self._templates.keys())
            raise ValueError(
                f"Template '{template_name}' not found.\n"
                f"Available templates: {available}\n"
                f"Use builder.list_templates() to see details."
            )

        template = self._templates[template_name]

        # Deep-copy so repeated calls don't mutate the template
        layer_specs = copy.deepcopy(template["layers"])

        layers = []
        for layer_spec in layer_specs:
            layer_type = layer_spec.pop("type")

            if layer_type == "dense":
                layers.append(self.Dense(**layer_spec))
            elif layer_type == "conv2d":
                layers.append(self.Conv2D(**layer_spec))
            elif layer_type == "dropout":
                layers.append(self.Dropout(**layer_spec))
            elif layer_type == "maxpool2d":
                layers.append(self.MaxPooling2D(**layer_spec))
            elif layer_type == "flatten":
                layers.append(self.Flatten())
            elif layer_type == "batchnorm":
                layers.append(self.BatchNorm())

        model = self.Sequential(layers, name=template_name)
        model.template_info = template

        return model

    def list_templates(self):
        """Show all available model templates with descriptions."""
        print("\n🏗️  Available Model Templates:\n")

        for name, template in self._templates.items():
            print(f"  📦 {name}")
            print(f"     {template['description']}")
            print(f"     Good for: {', '.join(template['good_for'])}")
            print()

    def suggest_architecture(
        self,
        task: str,
        input_shape: tuple,
        output_size: int,
    ):
        """
        Get architecture suggestions based on your task.

        Args:
            task: 'classification', 'regression', 'image_classification', etc.
            input_shape: Shape of your input data
            output_size: Number of outputs

        Examples:
            >>> builder = ModelBuilder()
            >>> builder.suggest_architecture(
            ...     task="image_classification",
            ...     input_shape=(28, 28, 1),
            ...     output_size=10
            ... )
        """
        print(f"\n💡 Architecture Suggestions for {task}:\n")

        if task in ("classification", "image_classification"):
            if len(input_shape) == 3:
                print("   Recommended: Convolutional Neural Network (CNN)")
                print(
                    "   Why: CNNs are designed to detect patterns in images"
                )
                print("\n   Suggested architecture:")
                print("   1. Conv2D(32, kernel_size=3) + ReLU")
                print("   2. MaxPooling2D()")
                print("   3. Conv2D(64, kernel_size=3) + ReLU")
                print("   4. MaxPooling2D()")
                print("   5. Flatten()")
                print("   6. Dense(128) + ReLU")
                print("   7. Dropout(0.3)")
                print(f"   8. Dense({output_size}) + Softmax")
                print("\n   Use: builder.from_template('image_classifier')")
            else:
                print("   Recommended: Simple Feedforward Network")
                print("   Why: Efficient for structured/tabular data")
                print("\n   Suggested architecture:")
                print("   1. Dense(128) + ReLU")
                print("   2. Dropout(0.2)")
                print("   3. Dense(64) + ReLU")
                print(f"   4. Dense({output_size}) + Softmax")
                print(
                    "\n   Use: builder.from_template('simple_classifier')"
                )

        elif task == "regression":
            print("   Recommended: Regression Network")
            print("   Why: Predicts continuous values")
            print("\n   Suggested architecture:")
            print("   1. Dense(64) + ReLU")
            print("   2. Dense(32) + ReLU")
            print("   3. Dense(1) + Linear (no activation)")
            print("\n   Use: builder.from_template('regression')")

        elif task == "binary_classification":
            print("   Recommended: Binary Classifier")
            print("   Why: Two-class output with sigmoid")
            print("\n   Suggested architecture:")
            print("   1. Dense(64) + ReLU")
            print("   2. Dropout(0.3)")
            print("   3. Dense(32) + ReLU")
            print("   4. Dense(1) + Sigmoid")
            print(
                "\n   Use: builder.from_template('binary_classifier')"
            )

        print(
            "\n   💡 Tip: Start with these suggestions, then experiment!"
        )


class Model:
    """
    Represents a complete neural network model.

    Educational model class that provides beginner-friendly interfaces
    for understanding, compiling, training, and evaluating models.
    """

    def __init__(
        self,
        layers: List[Layer],
        name: Optional[str] = None,
        craft=None,
    ):
        """
        Initialize a Model.

        Args:
            layers: List of Layer instances
            name: Optional model name
            craft: NeuroCraft instance
        """
        self.layers = layers
        self.name = name or "UntitledModel"
        self.craft = craft
        self.template_info: Optional[Dict] = None

        # Training state
        self.is_compiled = False
        self.loss_function = None
        self.loss_name: Optional[str] = None
        self.optimizer: Optional[str] = None
        self.learning_rate: float = 0.001
        self.metrics: List[str] = []

        # History
        self.history: Dict[str, List[float]] = {
            "loss": [],
            "accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }

    def _get_craft(self):
        """Lazily initialize craft if needed."""
        if self.craft is None:
            from neurogebra.core.neurocraft import NeuroCraft

            self.craft = NeuroCraft(educational_mode=False)
        return self.craft

    def summary(self, educational: bool = True):
        """
        Print model architecture summary.

        Args:
            educational: Include educational explanations for each layer
        """
        print(f"\n{'='*60}")
        print(f"  Model: {self.name}")
        print(f"{'='*60}\n")

        total_params = 0
        prev_units: Optional[int] = None

        for i, layer in enumerate(self.layers, 1):
            print(f"  Layer {i}: {layer}")

            if educational:
                print(f"           {layer.description}")

            # Estimate parameters (simplified)
            if layer.layer_type == "dense" and layer.units:
                input_size = prev_units if prev_units else 100
                params = input_size * layer.units + layer.units  # W + bias
                total_params += params
                print(f"           Parameters: ~{params:,}")

            if layer.units is not None:
                prev_units = layer.units

            print()

        print(f"{'='*60}")
        print(f"  Total Parameters: ~{total_params:,}")
        print(f"{'='*60}\n")

        if educational and self.template_info:
            print("  📚 About this architecture:")
            print(f"     {self.template_info['description']}")
            print(
                f"     Good for: {', '.join(self.template_info['good_for'])}"
            )
            print()

    def explain_architecture(self):
        """Provide a detailed explanation of every layer."""
        print(f"\n🎓 Understanding Your Model: {self.name}\n")

        for i, layer in enumerate(self.layers, 1):
            print(f"{'='*60}")
            print(f"Layer {i}: {layer.layer_type.upper()}")
            print(f"{'='*60}")
            layer.explain()

            if i < len(self.layers):
                print("\n   ⬇️  Passes data to next layer")
            print()

    def compile(
        self,
        loss: str = "mse",
        optimizer: str = "adam",
        metrics: Optional[List[str]] = None,
        learning_rate: float = 0.001,
    ):
        """
        Configure the model for training.

        Args:
            loss: Loss function name (e.g. 'mse', 'mae',
                  'binary_crossentropy')
            optimizer: Optimizer name ('adam', 'sgd', 'rmsprop')
            metrics: List of metrics to track (e.g. ['accuracy'])
            learning_rate: Learning rate for optimizer

        Examples:
            >>> model.compile(loss="mse", optimizer="adam",
            ...               metrics=["accuracy"])
        """
        print("\n⚙️  Compiling model...")

        self.loss_name = loss
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.metrics = metrics or ["accuracy"]

        # Try to get loss function from NeuroCraft
        try:
            craft = self._get_craft()
            self.loss_function = craft.get(loss)
        except KeyError:
            # Store name; trainer will handle it
            self.loss_function = None

        print(f"   Loss: {loss}")
        print(f"   Optimizer: {optimizer} (lr={learning_rate})")
        print(f"   Metrics: {', '.join(self.metrics)}")

        self.is_compiled = True
        print("   ✅ Model compiled successfully!\n")

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 10,
        batch_size: int = 32,
        validation_split: float = 0.2,
        verbose: bool = True,
        visualize: bool = False,
    ) -> Dict[str, List[float]]:
        """
        Train the model on data.

        Args:
            X: Input data
            y: Target data
            epochs: Number of training epochs
            batch_size: Batch size
            validation_split: Fraction for validation
            verbose: Print training progress
            visualize: Show training visualization

        Returns:
            Training history dictionary

        Examples:
            >>> model.fit(X_train, y_train, epochs=10)
        """
        if not self.is_compiled:
            raise RuntimeError(
                "Model must be compiled before training!\n"
                "Call model.compile() first."
            )

        print(f"\n🚀 Training {self.name}...\n")
        print(f"   Training samples: {len(X)}")
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {batch_size}\n")

        from neurogebra.training.educational_trainer import (
            EducationalTrainer,
        )

        trainer = EducationalTrainer(
            model=self,
            verbose=verbose,
            visualize=visualize,
        )

        history = trainer.train(
            X=X,
            y=y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
        )

        self.history = history

        print("\n✅ Training complete!")
        print(f"   Final loss: {history['loss'][-1]:.4f}")
        if history.get("accuracy"):
            print(f"   Final accuracy: {history['accuracy'][-1]:.4f}")

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.

        Args:
            X: Input data array

        Returns:
            Predictions as numpy array
        """
        print(f"🔮 Making predictions on {len(X)} samples...")
        # Placeholder - actual implementation uses backend
        output_units = 10
        for layer in reversed(self.layers):
            if layer.units is not None:
                output_units = layer.units
                break
        return np.random.rand(len(X), output_units)

    def evaluate(
        self, X: np.ndarray, y: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate model on test data.

        Args:
            X: Test input data
            y: Test target data

        Returns:
            Dictionary with loss and metric values
        """
        print(f"\n📊 Evaluating model on {len(X)} samples...")

        predictions = self.predict(X)
        loss = float(np.mean((predictions.flatten()[:len(y)] - y.flatten()[:len(predictions.flatten())]) ** 2))
        accuracy = float(
            np.mean(
                np.argmax(predictions, axis=-1)[:len(y)]
                == y.flatten()[:len(predictions)]
            )
        ) if predictions.ndim > 1 else 0.0

        print(f"   Test loss: {loss:.4f}")
        print(f"   Test accuracy: {accuracy:.4f}")

        return {"loss": loss, "accuracy": accuracy}

    def save(self, filepath: str):
        """Save model to file."""
        import pickle

        with open(filepath, "wb") as f:
            pickle.dump(
                {
                    "name": self.name,
                    "layers": [
                        {
                            "type": l.layer_type,
                            "units": l.units,
                            "activation": l.activation,
                            "params": l.params,
                        }
                        for l in self.layers
                    ],
                    "history": self.history,
                    "loss_name": self.loss_name,
                    "optimizer": self.optimizer,
                    "learning_rate": self.learning_rate,
                },
                f,
            )
        print(f"💾 Model saved to {filepath}")

    def plot_history(self):
        """Plot training history."""
        from neurogebra.viz.plotting import plot_training_history

        plot_training_history(self.history)

    def to_pytorch(self):
        """Export to PyTorch model."""
        from neurogebra.bridges.pytorch_bridge import to_pytorch

        return to_pytorch(self)

    def to_tensorflow(self):
        """Export to TensorFlow model."""
        from neurogebra.bridges.tensorflow_bridge import to_tensorflow

        return to_tensorflow(self)

    def __repr__(self):
        return (
            f"Model(name='{self.name}', layers={len(self.layers)}, "
            f"compiled={self.is_compiled})"
        )
