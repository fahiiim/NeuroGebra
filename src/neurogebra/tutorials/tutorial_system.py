"""
Interactive tutorial system for learning ML/DL concepts.

Provides step-by-step guided tutorials covering neural network basics,
building models, activations, and training.
"""

from typing import Optional


class TutorialSystem:
    """
    Interactive step-by-step tutorials for learning.

    Tutorials cover:
    - Basics of neural networks
    - Building your first model
    - Understanding activations
    - Choosing loss functions
    - Training and evaluation
    """

    def __init__(self):
        """Initialize tutorial system with available tutorials."""
        self.tutorials = {
            "basics": BasicsTutorial(),
            "first_model": FirstModelTutorial(),
            "activations": ActivationsTutorial(),
            "training": TrainingTutorial(),
        }

    def show_menu(self):
        """Show the tutorial menu."""
        print("\n" + "=" * 60)
        print("🎓 Neurogebra Interactive Tutorials")
        print("=" * 60 + "\n")

        print("Available tutorials:\n")
        print("1. 📚 basics          - Neural Network Basics (15 min)")
        print(
            "   Learn what neural networks are and how they work\n"
        )
        print(
            "2. 🏗️  first_model     - Build Your First Model (20 min)"
        )
        print("   Step-by-step guide to creating a classifier\n")
        print(
            "3. ⚡ activations      - Understanding Activations (15 min)"
        )
        print("   Explore different activation functions\n")
        print("4. 🎯 training        - Training Models (25 min)")
        print("   Learn how to train and evaluate models\n")

        print("=" * 60)
        print("\nTo start: craft.tutorial('basics')")
        print("=" * 60 + "\n")

    def start(self, topic: str):
        """
        Start a tutorial by topic name.

        Args:
            topic: Tutorial name ('basics', 'first_model',
                   'activations', 'training')
        """
        if topic not in self.tutorials:
            print(f"❌ Tutorial '{topic}' not found")
            self.show_menu()
            return

        tutorial = self.tutorials[topic]
        tutorial.run()


class BasicsTutorial:
    """Tutorial on neural network basics."""

    def run(self):
        """Run the basics tutorial."""
        print("\n" + "=" * 60)
        print("📚 Neural Network Basics - Interactive Tutorial")
        print("=" * 60 + "\n")

        self._step1_intro()
        self._step2_neurons()
        self._step3_layers()
        self._step4_example()

        print("\n✅ Tutorial complete! You're ready to build models.")
        print("   Next: Try craft.tutorial('first_model')\n")

    def _step1_intro(self):
        """Introduce neural networks."""
        print("-" * 60)
        print("STEP 1: What is a Neural Network?")
        print("-" * 60 + "\n")

        print("A neural network is like a team of workers:\n")
        print("  🧠 Input Layer:   Receives the data")
        print("         ↓")
        print("  🧠 Hidden Layers: Process and find patterns")
        print("         ↓")
        print("  🧠 Output Layer:  Makes final prediction")
        print()
        print("Each 'worker' (neuron) does a simple job:")
        print("  1. Takes inputs")
        print("  2. Multiplies by weights (learned values)")
        print("  3. Adds bias")
        print("  4. Applies activation function")
        print()

    def _step2_neurons(self):
        """Explain neurons with a concrete example."""
        print("-" * 60)
        print("STEP 2: How Neurons Work")
        print("-" * 60 + "\n")

        print("A neuron computes: output = activation(weights · inputs + bias)")
        print()

        inputs = [0.5, 0.8]
        weights = [0.4, 0.6]
        bias = 0.1

        weighted_sum = sum(i * w for i, w in zip(inputs, weights))
        import math

        output = 1 / (1 + math.exp(-(weighted_sum + bias)))

        print(f"  Inputs:       {inputs}")
        print(f"  Weights:      {weights}")
        print(f"  Weighted sum: {weighted_sum:.4f}")
        print(f"  Add bias:     {weighted_sum + bias:.4f}")
        print(f"  After sigmoid: {output:.4f}")
        print()

    def _step3_layers(self):
        """Explain layers."""
        print("-" * 60)
        print("STEP 3: Connecting Layers")
        print("-" * 60 + "\n")

        print("Neurons are organized in layers:\n")
        print("  Layer 1 (Input):  784 neurons (28x28 image)")
        print("      ↓ ↓ ↓ ↓ ↓")
        print("  Layer 2 (Hidden): 128 neurons")
        print("      ↓ ↓ ↓ ↓ ↓")
        print("  Layer 3 (Output): 10 neurons (digits 0-9)")
        print()
        print("Each connection has a weight that is learned during training!")
        print()

    def _step4_example(self):
        """Show a simple code example."""
        print("-" * 60)
        print("STEP 4: Try It Yourself!")
        print("-" * 60 + "\n")

        print("Build a tiny network with Neurogebra:\n")
        print("  >>> from neurogebra.builders import ModelBuilder")
        print("  >>> builder = ModelBuilder()")
        print("  >>> model = builder.Sequential([")
        print("  ...     builder.Dense(10, activation='relu'),")
        print("  ...     builder.Dense(5, activation='relu'),")
        print("  ...     builder.Dense(2, activation='softmax')")
        print("  ... ])")
        print("  >>> model.summary()")
        print()

        from neurogebra.builders.model_builder import ModelBuilder

        builder = ModelBuilder()
        model = builder.Sequential(
            [
                builder.Dense(10, activation="relu"),
                builder.Dense(5, activation="relu"),
                builder.Dense(2, activation="softmax"),
            ]
        )
        model.summary(educational=True)


class FirstModelTutorial:
    """Tutorial for building your first model."""

    def run(self):
        """Run the first model tutorial."""
        print("\n" + "=" * 60)
        print("🏗️  Build Your First Model - Interactive Tutorial")
        print("=" * 60 + "\n")

        print(
            "In this tutorial, you'll learn the steps to build a classifier!\n"
        )

        self._step1_problem()
        self._step2_architecture()
        self._step3_compile()
        self._step4_train()

        print("\n✅ Congratulations! You've learned the model-building workflow!")
        print("   Next: Experiment with different architectures\n")

    def _step1_problem(self):
        """Define the problem."""
        print("-" * 60)
        print("STEP 1: Understanding the Problem")
        print("-" * 60 + "\n")

        print("Task:   Recognize handwritten digits (0-9)")
        print("Data:   28x28 pixel images")
        print("Output: 10 classes (one for each digit)")
        print()
        print("This is a CLASSIFICATION problem")
        print()

    def _step2_architecture(self):
        """Design architecture."""
        print("-" * 60)
        print("STEP 2: Designing the Architecture")
        print("-" * 60 + "\n")

        print("For image classification, we'll use:\n")
        print("  1. Input:   784 neurons (28 x 28 pixels flattened)")
        print("  2. Hidden:  128 neurons with ReLU activation")
        print("  3. Dropout: 20% to prevent overfitting")
        print("  4. Hidden:  64 neurons with ReLU")
        print("  5. Output:  10 neurons with Softmax (one per class)")
        print()
        print("Code:")
        print("  >>> model = builder.Sequential([")
        print(
            "  ...     builder.Dense(128, activation='relu', "
            "input_shape=(784,)),"
        )
        print("  ...     builder.Dropout(0.2),")
        print("  ...     builder.Dense(64, activation='relu'),")
        print("  ...     builder.Dense(10, activation='softmax')")
        print("  ... ])")
        print()

    def _step3_compile(self):
        """Compile model."""
        print("-" * 60)
        print("STEP 3: Configuring for Training")
        print("-" * 60 + "\n")

        print("Before training, we need to specify:\n")
        print("  • Loss function: How to measure errors")
        print("    → 'mse' for regression, 'binary_crossentropy' for binary")
        print()
        print("  • Optimizer: How to update weights")
        print("    → 'adam' (adaptive learning rate, recommended)")
        print()
        print("  • Metrics: What to track")
        print("    → 'accuracy' (% correct predictions)")
        print()
        print("Code:")
        print("  >>> model.compile(")
        print("  ...     loss='mse',")
        print("  ...     optimizer='adam',")
        print("  ...     metrics=['accuracy']")
        print("  ... )")
        print()

    def _step4_train(self):
        """Describe training."""
        print("-" * 60)
        print("STEP 4: Training the Model")
        print("-" * 60 + "\n")

        print("Train the model on data:\n")
        print("  >>> model.fit(")
        print("  ...     X_train, y_train,")
        print("  ...     epochs=10,")
        print("  ...     batch_size=32,")
        print("  ...     validation_split=0.2")
        print("  ... )")
        print()
        print("This will:")
        print("  1. Show progress for each epoch")
        print("  2. Display live training graphs (if visualize=True)")
        print("  3. Provide tips if issues arise")
        print("  4. Save training history")
        print()


class ActivationsTutorial:
    """Tutorial on activation functions."""

    def run(self):
        """Run the activations tutorial."""
        print("\n" + "=" * 60)
        print("⚡ Understanding Activation Functions")
        print("=" * 60 + "\n")

        self._step1_what()
        self._step2_common()
        self._step3_choosing()

        print("\n✅ Tutorial complete!")
        print("   Try: craft.compare(['relu', 'sigmoid', 'tanh'])")
        print()

    def _step1_what(self):
        """What are activation functions?"""
        print("-" * 60)
        print("STEP 1: What Are Activation Functions?")
        print("-" * 60 + "\n")

        print(
            "Activation functions add NON-LINEARITY to neural networks."
        )
        print("Without them, a deep network would just be a linear model!\n")
        print("They decide which neurons 'fire' (output a signal).")
        print()

    def _step2_common(self):
        """Common activations."""
        print("-" * 60)
        print("STEP 2: Common Activation Functions")
        print("-" * 60 + "\n")

        activations_info = [
            ("ReLU", "max(0, x)", "Hidden layers (default choice)"),
            ("Sigmoid", "1/(1+e^-x)", "Binary output (0 to 1)"),
            ("Tanh", "tanh(x)", "Hidden layers (-1 to 1)"),
            ("Softmax", "e^xi / Σe^xj", "Multi-class output"),
            ("GELU", "x·Φ(x)", "Transformers, BERT"),
            ("Swish", "x·σ(x)", "Modern deep networks"),
        ]

        print(f"  {'Name':<12} {'Formula':<20} {'Best For'}")
        print("  " + "-" * 55)
        for name, formula, best_for in activations_info:
            print(f"  {name:<12} {formula:<20} {best_for}")
        print()

    def _step3_choosing(self):
        """How to choose."""
        print("-" * 60)
        print("STEP 3: How to Choose?")
        print("-" * 60 + "\n")

        print("Quick guide:\n")
        print("  Hidden layers → Start with ReLU")
        print("  Binary output → Sigmoid")
        print("  Multi-class   → Softmax")
        print("  Better ReLU   → Try GELU or Swish")
        print("  RNN/LSTM      → Tanh")
        print()
        print("  💡 When in doubt, use ReLU for hidden layers!")
        print()


class TrainingTutorial:
    """Tutorial on model training."""

    def run(self):
        """Run the training tutorial."""
        print("\n" + "=" * 60)
        print("🎯 Training Models - Interactive Tutorial")
        print("=" * 60 + "\n")

        self._step1_overview()
        self._step2_loss()
        self._step3_optimizers()
        self._step4_hyperparams()
        self._step5_evaluation()

        print("\n✅ Tutorial complete! You understand model training.")
        print("   Try building and training a model now!\n")

    def _step1_overview(self):
        """Training overview."""
        print("-" * 60)
        print("STEP 1: How Training Works")
        print("-" * 60 + "\n")

        print("Training is an iterative process:\n")
        print("  1. FORWARD PASS:  Model makes predictions")
        print("  2. LOSS:          Measure how wrong predictions are")
        print("  3. BACKWARD PASS: Compute gradients (how to improve)")
        print("  4. UPDATE:        Adjust weights to reduce loss")
        print("  5. REPEAT:        Do this for many epochs")
        print()

    def _step2_loss(self):
        """Loss functions."""
        print("-" * 60)
        print("STEP 2: Loss Functions")
        print("-" * 60 + "\n")

        print("The loss function measures prediction errors:\n")
        print("  MSE (Mean Squared Error)")
        print("    → For regression (continuous values)")
        print("    → Formula: mean((predicted - actual)²)")
        print()
        print("  Cross-Entropy")
        print("    → For classification (categories)")
        print("    → Measures probability distribution difference")
        print()
        print("  MAE (Mean Absolute Error)")
        print("    → For regression, robust to outliers")
        print("    → Formula: mean(|predicted - actual|)")
        print()

    def _step3_optimizers(self):
        """Optimizers."""
        print("-" * 60)
        print("STEP 3: Optimizers")
        print("-" * 60 + "\n")

        print("Optimizers decide HOW to update weights:\n")
        print("  SGD (Stochastic Gradient Descent)")
        print("    → Simple, well-understood")
        print("    → Needs careful learning rate tuning")
        print()
        print("  Adam (Adaptive Moment Estimation)")
        print("    → Adapts learning rate per parameter")
        print("    → Recommended default for most tasks")
        print()
        print("  RMSprop")
        print("    → Good for RNNs and non-stationary problems")
        print()
        print("  💡 Start with Adam, learning_rate=0.001")
        print()

    def _step4_hyperparams(self):
        """Hyperparameters."""
        print("-" * 60)
        print("STEP 4: Key Hyperparameters")
        print("-" * 60 + "\n")

        print("Hyperparameters YOU choose:\n")
        print(
            "  Learning Rate (0.001): How big each weight update is"
        )
        print("    Too high → unstable, too low → very slow")
        print()
        print("  Batch Size (32): Samples per gradient update")
        print("    Larger → faster but more memory")
        print()
        print("  Epochs (10-100): How many times to see all data")
        print("    Too many → overfitting, too few → underfitting")
        print()
        print(
            "  Dropout Rate (0.2): Fraction of neurons to randomly disable"
        )
        print("    Helps prevent overfitting")
        print()

    def _step5_evaluation(self):
        """Evaluation."""
        print("-" * 60)
        print("STEP 5: Evaluating Your Model")
        print("-" * 60 + "\n")

        print("After training, evaluate on TEST data:\n")
        print("  >>> results = model.evaluate(X_test, y_test)")
        print()
        print("Watch for:")
        print("  ✅ Training loss decreasing steadily")
        print("  ✅ Validation loss following training loss")
        print("  ⚠️  Val loss much higher → OVERFITTING")
        print("  ⚠️  Both losses high    → UNDERFITTING")
        print()
        print("  💡 If overfitting: add dropout, get more data")
        print("  💡 If underfitting: bigger model, train longer")
        print()
