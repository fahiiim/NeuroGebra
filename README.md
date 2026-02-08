<p align="center">
  <strong>Neurogebra</strong>
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

Neurogebra is a unified Python library that bridges symbolic mathematics, numerical computation, and machine learning. It provides **285 pre-built, tested, and documented mathematical expressions** spanning activations, losses, statistics, optimization, linear algebra, and more — each one symbolic, numerically evaluable, trainable, and accompanied by educational metadata.

Unlike traditional ML frameworks, Neurogebra is designed as a **mathematical formula companion**: a searchable, executable encyclopedia of the formulas that power modern AI, with built-in explanations, gradient computation, and composition tools.

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Building and Training Models](#building-and-training-models)
- [Symbolic Gradients and Composition](#symbolic-gradients-and-composition)
- [Autograd Engine](#autograd-engine)
- [Search and Discovery](#search-and-discovery)
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
pip install neurogebra[docs]       # Documentation tools
pip install neurogebra[dev]        # Development and testing tools
pip install neurogebra[all]        # Everything
```

**Requirements:** Python 3.9+ | NumPy | SymPy | Matplotlib | SciPy

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
