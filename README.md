<p align="center">
  <img src="https://raw.githubusercontent.com/fahiiim/NeuroGebra/main/assets/logo.png" alt="Neurogebra Logo" width="600">
</p>

<p align="center">
  <em>The executable mathematical formula companion for AI and Data Science</em>
</p>

<p align="center">
  <a href="https://github.com/fahiiim/NeuroGebra/actions/workflows/tests.yml"><img src="https://github.com/fahiiim/NeuroGebra/actions/workflows/tests.yml/badge.svg" alt="Tests"></a>
  <a href="https://pypi.org/project/neurogebra/"><img src="https://img.shields.io/pypi/v/neurogebra.svg" alt="PyPI version"></a>
  <a href="https://pypi.org/project/neurogebra/"><img src="https://img.shields.io/pypi/pyversions/neurogebra.svg" alt="Python versions"></a>
  <a href="https://codecov.io/github/fahiiim/NeuroGebra"><img src="https://codecov.io/github/fahiiim/NeuroGebra/graph/badge.svg?token=E819QI1LO0" alt="codecov"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT"></a>
  <a href="https://pypi.org/project/neurogebra/"><img src="https://img.shields.io/pypi/dm/neurogebra.svg" alt="Downloads"></a>
  <a href="https://neurogebra.readthedocs.io"><img src="https://img.shields.io/readthedocs/neurogebra.svg" alt="Docs"></a>
  <a href="https://github.com/fahiiim/NeuroGebra"><img src="https://img.shields.io/github/stars/fahiiim/NeuroGebra?style=social" alt="GitHub stars"></a>
</p>

---

Neurogebra is a unified Python library that bridges symbolic mathematics, numerical computation, and machine learning. It provides **285 pre-built, tested, and documented mathematical expressions** spanning activations, losses, statistics, optimization, linear algebra, and more — each one symbolic, numerically evaluable, trainable, and accompanied by educational metadata. Additionally, it includes **100+ curated datasets** for immediate experimentation and learning.

Unlike traditional ML frameworks, Neurogebra is designed as a **mathematical formula companion**: a searchable, executable encyclopedia of the formulas that power modern AI, with built-in explanations, gradient computation, composition tools, and ready-to-use datasets.

### What's New in v1.2.1 — Training Observatory 🔭

> **See every neuron fire. Watch every gradient flow. Understand every weight update — in colour.**

The Training Observatory is an advanced training logging and visualization system that brings unprecedented mathematical transparency to neural network training. It performs **real forward/backward computation** through every layer and displays the mathematics in colourful, depth-level detail right in your terminal.

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training Observatory (v1.2.1)](#training-observatory-v121)
- [Complete Feature List](#complete-feature-list)
- [Building and Training Models](#building-and-training-models)
- [Symbolic Gradients and Composition](#symbolic-gradients-and-composition)
- [Autograd Engine](#autograd-engine)
- [Search and Discovery](#search-and-discovery)
- [Datasets](#datasets)
- [Expression Library](#expression-library)
- [Architecture](#architecture)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)
- [Positioning: What Neurogebra Is and Is Not](#positioning-what-neurogebra-is-and-is-not)

---

## Installation

```bash
pip install neurogebra
```

Optional extras for extended functionality:

```bash
pip install neurogebra[viz]        # Interactive visualization (Plotly)
pip install neurogebra[frameworks] # PyTorch, TensorFlow bridges
pip install neurogebra[logging]    # TensorBoard & W&B integration
pip install neurogebra[docs]       # Documentation tools
pip install neurogebra[dev]        # Development and testing tools
pip install neurogebra[all]        # Everything
```

**Requirements:** Python 3.9+ | NumPy | SymPy | Matplotlib | SciPy | Rich | Colorama

---

## Quick Start

```python
from neurogebra import MathForge

forge = MathForge()

# Retrieve a pre-built activation function
relu = forge.get("relu")
print(relu.eval(x=5))       # 5
print(relu.eval(x=-3))      # 0
print(relu.formula)          # LaTeX representation
print(relu.explain())        # Plain-language explanation

# Retrieve any of the 285 expressions by name
adam = forge.get("adam_step")
gaussian = forge.get("gaussian")
f1 = forge.get("f1_score_formula")
```

---

## Training Observatory (v1.2.1)

The **Training Observatory** is a first-of-its-kind training logging system that shows you the complete mathematical picture of what happens inside your neural network during training — **in real time, in colour, in your terminal**.

### Instant Setup — One Argument

```python
from neurogebra.builders.model_builder import ModelBuilder

builder = ModelBuilder()
model = builder.Sequential([
    builder.Dense(64, activation="relu"),
    builder.Dense(32, activation="tanh"),
    builder.Dense(1, activation="sigmoid"),
], name="my_model")

# Just add log_level to compile()
model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    learning_rate=0.01,
    log_level="expert",          # ← this is all you need
)

model.fit(X_train, y_train, epochs=20, batch_size=32)
```

### What You'll See

The Observatory produces colourful, structured output directly in your terminal:

- 🟢 **Green** — Healthy metrics (loss decreasing, gradients stable)
- 🟡 **Yellow** — Warnings (high gradient variance, saturation starting)
- 🔴 **Red** — Danger (vanishing/exploding gradients, NaN detected)
- 🟣 **Purple/Magenta** — Mathematical formulas (forward/backward equations)
- 🔵 **Blue** — Informational messages (epoch/batch progress)

### Log Levels

| Level | What You See |
|-------|-------------|
| `"basic"` | Epoch-level loss and accuracy, start/end messages |
| `"detailed"` | + Batch-level progress, timing information |
| `"expert"` | + Layer-by-layer formulas, gradient norms, weight stats |
| `"debug"` | + Every tensor shape, raw statistics, full computation trace |

### Preset Configurations

```python
from neurogebra.logging.config import LogConfig

# Choose a preset
config = LogConfig.minimal()     # Just epoch progress
config = LogConfig.standard()    # Layer info + timing + health checks
config = LogConfig.verbose()     # Full math depth — every formula, every gradient
config = LogConfig.research()    # Everything + export to files

model.compile(
    loss="mse",
    optimizer="adam",
    log_config=config,
)
```

### Layer-by-Layer Mathematical Formulas

At **expert** level, the Observatory shows the exact computation happening inside each layer:

```
Forward:  a₁ = relu(W₁·x + b₁)    │ shape: (32, 64) → (32, 32)
Forward:  a₂ = tanh(W₂·a₁ + b₂)   │ shape: (32, 32) → (32, 16)
Forward:  ŷ  = σ(W₃·a₂ + b₃)      │ shape: (32, 16) → (32, 1)

Backward: ∂L/∂W₃ = ∂L/∂ŷ ⊙ σ'(z₃) · a₂ᵀ
Backward: ∂L/∂W₂ = ∂L/∂a₂ ⊙ tanh'(z₂) · a₁ᵀ
Backward: ∂L/∂W₁ = ∂L/∂a₁ ⊙ relu'(z₁) · xᵀ
```

### Smart Health Diagnostics

The Observatory automatically detects problems and provides actionable recommendations:

```
🚨 [CRITICAL] NaN/Inf Detected
   NaN values found in training loss!
   → Check for division by zero in your data
   → Reduce learning rate (try 1e-4)
   → Add gradient clipping

⚠️  [WARNING] Overfitting Detected
   Validation loss increasing while training loss decreases (ratio: 1.8×)
   → Add dropout layers (rate 0.2-0.5)
   → Reduce model complexity
   → Increase training data or use data augmentation
   → Try early stopping

🔴 [DANGER] Vanishing Gradients
   Layer 'dense_3' gradient L2 norm = 2.1e-09
   → Use ReLU/LeakyReLU instead of sigmoid/tanh
   → Add batch normalization
   → Use skip connections
```

### Export Training Logs

```python
config = LogConfig.research()
config.export_formats = ["json", "csv", "html", "markdown"]
config.export_dir = "./my_training_logs"

model.compile(loss="mse", optimizer="adam", log_config=config)
model.fit(X, y, epochs=50)

# After training, check ./my_training_logs/ for:
#   training_log.json   — Full structured event log
#   metrics.csv         — Epoch-level metrics table
#   report.html         — Interactive HTML report with Chart.js graphs
#   report.md           — Human-readable Markdown report
```

### Standalone Monitors

Use the monitoring tools independently, outside of model training:

```python
from neurogebra.logging.monitors import GradientMonitor, WeightMonitor
from neurogebra.logging.health_checks import SmartHealthChecker
import numpy as np

# Monitor gradients
gm = GradientMonitor()
stats = gm.record("layer_0", np.random.randn(64, 32) * 0.001)
print(stats["status"])   # "healthy", "danger", or "critical"
print(stats["alerts"])   # Human-readable alerts if any

# Run health checks on your training history
checker = SmartHealthChecker()
alerts = checker.run_all(
    epoch=10,
    train_losses=[1.0, 0.8, 0.5, 0.3, 0.15],
    val_losses=[1.0, 0.9, 0.85, 0.95, 1.1],
    gradient_norms={"dense_0": 0.5, "dense_1": 1e-9},
)
for alert in alerts:
    print(f"[{alert.severity}] {alert.message}")
    for rec in alert.recommendations:
        print(f"  → {rec}")
```

---

## Complete Feature List

### Core Mathematical Engine
- **285 Pre-built Expressions** — Activations, losses, metrics, optimizers, statistics, transforms, and more
- **Symbolic Mathematics** — Full SymPy integration with LaTeX rendering
- **Fast Numerical Evaluation** — NumPy-backed lambdify for production-speed computation
- **Analytical Gradients** — Symbolic differentiation for any expression
- **Expression Composition** — Combine expressions with arithmetic (+, -, *, /) and function composition
- **Trainable Parameters** — Attach learnable coefficients to any symbolic expression
- **Searchable Repository** — Find formulas by name, category, keyword, or use case

### Neural Network Builder
- **ModelBuilder API** — Intuitive, Keras-like interface for building neural networks
- **Layer Types** — Dense, Conv2D, Dropout, BatchNorm, MaxPooling2D, Flatten
- **Architecture Templates** — Pre-built templates for classification, regression, image recognition, binary classification
- **Educational Explanations** — Every layer comes with plain-language descriptions and "explain()" methods
- **Real Forward/Backward Passes** — Actual matrix multiplications, gradient computation (not simulated)
- **Model Summary & Architecture Visualization** — See parameter counts and data flow

### Training Observatory (v1.2.1) 🔭
- **Colour-coded Terminal Display** — Green (healthy), yellow (warning), red (danger), purple (formulas)
- **5 Log Levels** — Silent, Basic, Detailed, Expert, Debug
- **Layer-by-Layer Formula Display** — Forward and backward pass equations in Unicode math notation
- **Gradient Flow Monitoring** — Vanishing/exploding gradient detection with L1/L2 norms
- **Weight Distribution Tracking** — Histogram, dead neuron detection, weight change percentage
- **Activation Monitoring** — Dead ReLU detection, sigmoid/tanh saturation analysis
- **Per-Layer Timing** — Identify computational bottlenecks
- **Smart Health Diagnostics** — Automatic detection of 8+ training problems with actionable recommendations
- **Computation Graph Tracking** — Full DAG of operations with shapes, values, and gradients
- **4 Export Formats** — JSON (structured), CSV (metrics), HTML (interactive charts), Markdown (reports)
- **Preset Configurations** — Minimal, Standard, Verbose, Research, Production
- **Standalone Monitors** — Use gradient/weight/activation monitors independently
- **Formula Renderer** — Unicode and LaTeX rendering of forward/backward formulas, loss functions
- **Image Logger** — ASCII art rendering of input images and activation maps in terminal

### Autograd Engine
- **From-Scratch Automatic Differentiation** — Educational autograd with Value and Tensor classes
- **Computation Graph** — Build and inspect the gradient computation graph
- **Backpropagation** — Automatic gradient computation through arbitrary expressions
- **Operations** — Add, multiply, power, relu, sigmoid, tanh, exp, log, and more

### Datasets (100+)
- **Classification** — Iris, Wine, Breast Cancer, MNIST, Fashion-MNIST, Spam, Titanic, Adult Income
- **Regression** — California Housing, Diabetes, Auto MPG, Bike Sharing, Energy Efficiency
- **Synthetic Patterns** — XOR, Moons, Circles, Spirals, Checkerboard, Blobs, Swiss Roll
- **Time Series** — Sine Waves, Random Walks, Stock Prices, Seasonal Data
- **Image Recognition** — MNIST, Fashion-MNIST, Digits (8×8), CIFAR-style
- **Text/NLP** — Spam Detection, Sentiment Analysis
- **Educational Metadata** — Difficulty level, use cases, sample count for every dataset

### Framework Bridges
- **PyTorch Export** — Convert Neurogebra models to PyTorch `nn.Module`
- **TensorFlow Export** — Convert to TensorFlow/Keras models
- **Seamless Workflow** — Prototype in Neurogebra, deploy in production frameworks

### Visualization
- **Training History Plots** — Loss and accuracy curves with Matplotlib
- **Expression Visualization** — Plot any mathematical expression
- **Interactive Plots** — Plotly-based interactive visualization (optional)

### Educational Features
- **NeuroCraft** — Educational interface with guided tutorials and explanations
- **Interactive Tutorials** — Step-by-step lessons on tensors, gradients, training, and more
- **Explain Everything** — Every expression, layer, and operation has an `.explain()` method
- **Educational Trainer** — Training with real-time tips, debugging advice, and step explanations
- **Plain-Language Descriptions** — No jargon — every concept explained for beginners

---

## Building and Training Models

Neurogebra includes a model builder and educational trainer designed for clarity and learning.

```python
from neurogebra.builders.model_builder import ModelBuilder
from neurogebra.training.educational_trainer import EducationalTrainer
from neurogebra.datasets.loaders import Datasets
import numpy as np

# Load a dataset
X, y = Datasets.load_moons(n_samples=500, noise=0.2)

# Build a model using the builder API
builder = ModelBuilder()
model = builder.Sequential([
    builder.Dense(64, activation="relu", input_shape=(2,)),
    builder.Dropout(0.2),
    builder.Dense(32, activation="relu"),
    builder.Dense(1, activation="sigmoid")
], name="moon_classifier")

# Inspect the architecture
model.summary()
model.explain_architecture()

# Compile with loss and optimizer
model.compile(optimizer="adam", loss="binary_crossentropy", learning_rate=0.01)

# Train with the educational trainer (provides real-time tips and debugging advice)
trainer = EducationalTrainer(model, verbose=True, explain_steps=True)
history = trainer.train(X, y, epochs=20, batch_size=32, validation_split=0.2)
```

You can also **train symbolic expressions directly** — formulas with learnable parameters:

```python
from neurogebra import Expression
from neurogebra.core.trainer import Trainer
import numpy as np

# Define a trainable expression: y = m*x + b
expr = Expression(
    "fit_line",
    "m*x + b",
    params={"m": 0.0, "b": 0.0},
    trainable_params=["m", "b"]
)

# Generate data: y = 2x + 1
X = np.linspace(0, 10, 100)
y = 2 * X + 1 + np.random.normal(0, 0.5, 100)

# Train
trainer = Trainer(expr, learning_rate=0.01, optimizer="adam")
history = trainer.fit(X, y, epochs=200, verbose=True)

print(f"Learned: m={expr.params['m']:.2f}, b={expr.params['b']:.2f}")
# Output: Learned: m=2.00, b=1.01
```

---

## Symbolic Gradients and Composition

Every expression supports symbolic differentiation for analytical gradient inspection.

```python
from neurogebra import MathForge

forge = MathForge()

sigmoid = forge.get("sigmoid")

# Compute the symbolic derivative
sigmoid_grad = sigmoid.gradient("x")
print(sigmoid_grad.formula)   # Analytical derivative in LaTeX

# Compose expressions
mse = forge.get("mse")
mae = forge.get("mae")
custom_loss = 0.7 * mse + 0.3 * mae   # Weighted combination

# Evaluate the composed expression
print(custom_loss.eval(y=1.0, y_pred=0.8))
```

---

## Autograd Engine

Neurogebra includes a from-scratch automatic differentiation engine for understanding how backpropagation works internally.

```python
from neurogebra.core.autograd import Value

# Build a computation graph
x = Value(2.0)
w = Value(-3.0)
b = Value(1.0)

# Forward pass
y = w * x + b
z = y.relu()

# Backward pass
z.backward()

print(f"dz/dw = {w.grad}")   # 2.0
print(f"dz/dx = {x.grad}")   # -3.0
```

---

## Search and Discovery

```python
from neurogebra import MathForge

forge = MathForge()

# Search by keyword
results = forge.search("classification")

# List by category
forge.list_all(category="activation")
forge.list_all(category="loss")

# Compare multiple expressions side by side
forge.compare(["relu", "sigmoid", "tanh"])
```

---

## Datasets

Neurogebra includes **100+ curated datasets** for fast experimentation and learning.

```python
from neurogebra.datasets import Datasets

# Browse all available datasets
Datasets.list_all()

# Load a dataset
(X_train, y_train), (X_test, y_test) = Datasets.load_iris(verbose=True)

# Search for datasets by keyword
Datasets.search("classification")
Datasets.search("image")
Datasets.search("medical")

# Get detailed info
Datasets.get_info("california_housing")
```

### Dataset Categories

| Category | Count | Examples |
|----------|------:|----------|
| **Classification** | 25+ | Iris, Wine, Breast Cancer, MNIST, Fashion-MNIST, Spam, Titanic, Adult Income |
| **Regression** | 25+ | California Housing, Diabetes, Auto MPG, Bike Sharing, Energy Efficiency |
| **Synthetic Patterns** | 20+ | XOR, Moons, Circles, Spirals, Checkerboard, Blobs, Swiss Roll |
| **Time Series** | 15+ | Sine Waves, Random Walks, Stock Prices, Seasonal Data, AR Processes |
| **Image Recognition** | 10+ | MNIST, Fashion-MNIST, Digits (8x8), CIFAR-style |
| **Text/NLP** | 5+ | Spam Detection, Sentiment Analysis |

Every dataset includes:
- **Educational metadata** (difficulty, use cases, sample count)
- **Pre-split train/test** sets (where applicable)
- **Verbose mode** for learning what each dataset contains
- **Consistent interface** - all return numpy arrays ready for training

Use `ExpandedDatasets` to access 80+ additional specialized datasets for advanced topics.

---

## Expression Library

Neurogebra ships with 285 verified mathematical expressions organized into 10 domain modules:

| Module | Count | Scope |
|--------|------:|-------|
| Activations | 15 | ReLU, Sigmoid, Tanh, Swish, GELU, Mish, ELU, SELU, and more |
| Losses | 8 | MSE, MAE, Cross-Entropy, Huber, Hinge, Log-Cosh, Quantile |
| Regularizers | 20 | L1, L2, Elastic Net, Dropout, SCAD, MCP, Group Lasso, Tikhonov |
| Algebra | 48 | Polynomials, kernels, probability distributions, special functions |
| Calculus | 48 | Elementary, trigonometric, hyperbolic, Taylor series, integral transforms |
| Statistics | 35 | PDFs, CDFs, information theory, Bayesian inference, regression |
| Linear Algebra | 24 | Norms, distances, projections, matrix operations, attention mechanisms |
| Optimization | 27 | SGD, Adam, AdamW, learning rate schedules, loss landscapes |
| Metrics | 27 | Precision, Recall, F1, R-squared, AIC, BIC, NDCG, Matthews correlation |
| Transforms | 33 | Normalization, encoding, weight initialization, signal processing |

Every expression includes:
- **Symbolic representation** (SymPy) with LaTeX rendering
- **Fast numerical evaluation** (NumPy-backed via lambdify)
- **Gradient computation** (analytical, symbolic)
- **Educational metadata** (description, category, use cases, pros/cons)
- **Composability** (arithmetic operations, function composition)
- **Trainable parameters** (optional learnable coefficients)

---

## Architecture

```
neurogebra/
  core/
    expression.py     # Unified Expression class (symbolic + numerical + trainable)
    forge.py          # MathForge: central expression hub and search
    neurocraft.py     # NeuroCraft: educational interface with tutorials
    autograd.py       # Micro autograd engine (Value, Tensor)
    trainer.py        # Parameter optimization (SGD, Adam)
  repository/         # 10 domain modules, 285 expressions
  builders/           # ModelBuilder: architecture templates and guidance
  training/           # EducationalTrainer: training with explanations
  logging/            # 🔭 Training Observatory (v1.2.1)
    logger.py         #   Event-driven training logger
    config.py         #   Preset configurations
    monitors.py       #   Gradient, weight, activation, performance monitors
    health_checks.py  #   Smart diagnostics with recommendations
    terminal_display.py # Rich colour-coded terminal renderer
    formula_renderer.py # Unicode/LaTeX math formula display
    image_logger.py   #   ASCII pixel art for images & activations
    exporters.py      #   JSON, CSV, HTML, Markdown exporters
    computation_graph.py # Full DAG tracker
  tutorials/          # Interactive step-by-step tutorial system
  datasets/           # Built-in dataset loaders (MNIST, Iris, moons, etc.)
  bridges/            # Framework converters (PyTorch, TensorFlow, JAX)
  viz/                # Visualization tools (matplotlib, plotly)
  utils/              # Helpers and explanation engine
```

---

## Documentation

Full documentation: [neurogebra.readthedocs.io](https://neurogebra.readthedocs.io)

- [Getting Started](https://neurogebra.readthedocs.io/getting-started)
- [Tutorials](https://neurogebra.readthedocs.io/tutorials)
- [API Reference](https://neurogebra.readthedocs.io/api)

---

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
git clone https://github.com/fahiiim/NeuroGebra.git
cd NeuroGebra
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
pip install -e ".[dev]"
pytest tests/ -v
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Positioning: What Neurogebra Is and Is Not

### Neurogebra is not a competitor to TensorFlow or PyTorch.

TensorFlow and PyTorch are production-grade deep learning frameworks built for training large-scale neural networks on GPUs and TPUs. They are industry standards for model development, deployment, and research at scale. Neurogebra does not attempt to replace, replicate, or compete with them in any capacity.

### What Neurogebra actually is.

Neurogebra occupies a fundamentally different space. It is a **mathematical formula library with executable, symbolic, and educational capabilities**. The closest analogues are tools like Wolfram Mathematica (proprietary, expensive, not Python-native) or manually assembling formulas from SymPy and Wikipedia (no curation, no ML focus, no educational layer).

Neurogebra provides a unique combination that does not exist in any single tool today:

| Capability | Neurogebra | SymPy | NumPy | Mathematica | PyTorch / TF |
|---|:---:|:---:|:---:|:---:|:---:|
| Symbolic math (LaTeX, calculus) | Yes | Yes | No | Yes | No |
| Fast numerical evaluation | Yes | Slow | Yes | Yes | Yes |
| 285 curated ML/statistics formulas | **Yes** | No | No | No | No |
| Educational metadata per formula | **Yes** | No | No | No | No |
| Trainable symbolic parameters | **Yes** | No | No | No | N/A |
| Searchable formula repository | **Yes** | No | No | No | No |
| Free and open source | Yes | Yes | Yes | No | Yes |
| Python native | Yes | Yes | Yes | No | Yes |

### Why students should use this.

**It is an executable reference library, not a framework to master.**
Students do not need to "learn Neurogebra" the way they learn PyTorch. They use it to look up, verify, and experiment with mathematical formulas. The command `forge.get("adam_step")` immediately returns the Adam optimizer update rule as a symbolic expression with documentation attached — no textbook lookup required.

**It bridges the gap between mathematical theory and code.**
In most curricula, students learn formulas on a whiteboard and then separately implement them in code. Neurogebra collapses that gap: every formula is simultaneously a symbolic object (inspect the math), a numerical function (run it on data), and an educational resource (read what it does and when to use it).

**It eliminates transcription errors.**
Students routinely introduce bugs when translating formulas from papers or textbooks into code. Neurogebra's 285 expressions are verified with 377 automated tests. Using `forge.get("cross_entropy")` is faster and more reliable than re-deriving it from scratch.

**It complements existing tools rather than replacing them.**
Students prototype and verify formulas in Neurogebra, then implement production models in PyTorch or TensorFlow. The framework bridges (PyTorch, TF, JAX converters) explicitly support this workflow. Neurogebra is the scratchpad; PyTorch is the production line.

**It teaches through transparency.**
The autograd engine, the educational trainer with real-time debugging advice, the layer explanation system, and the interactive tutorials are designed to make invisible processes visible. Students do not just see numbers — they see why their loss is diverging, what each layer does, and how gradients flow through a computation graph.

---

<p align="center">
  <strong>Author:</strong> <a href="https://github.com/fahiiim">Fahim Sarker</a>
  <br>
  <a href="https://github.com/fahiiim/NeuroGebra">GitHub</a> &middot; <a href="https://pypi.org/project/neurogebra/">PyPI</a> &middot; <a href="https://neurogebra.readthedocs.io">Documentation</a>
</p>
