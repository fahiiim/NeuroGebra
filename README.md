# Neurogebra

**The executable mathematical formula companion for AI & Data Science**

[![Tests](https://github.com/fahiiim/NeuroGebra/actions/workflows/tests.yml/badge.svg)](https://github.com/fahiiim/NeuroGebra/actions/workflows/tests.yml)
[![PyPI version](https://badge.fury.io/py/neurogebra.svg)](https://badge.fury.io/py/neurogebra)
[![codecov](https://codecov.io/github/fahiiim/NeuroGebra/graph/badge.svg?token=E819QI1LO0)](https://codecov.io/github/fahiiim/NeuroGebra)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

Neurogebra is a unified Python library that bridges symbolic mathematics, numerical computation, and deep learning. **285 pre-built, tested, and documented mathematical expressions** — from activations and losses to statistics, optimization, and linear algebra — all symbolic, numerical, trainable, and educational.

## 🚀 Quick Start

```bash
pip install neurogebra
```

```python
from neurogebra import MathForge

forge = MathForge()

# Get a pre-built activation function
relu = forge.get("relu")
result = relu.eval(x=5)  # 5

# It's also callable
result = relu(x=-3)  # 0

# See the formula
print(relu.formula)  # LaTeX representation

# Get an explanation
print(relu.explain())
```

## ✨ Features

- **285 Pre-built Expressions**: Activations, losses, regularizers, algebra, calculus, statistics, linear algebra, optimization, metrics, and transforms
- **Symbolic + Numerical**: Every expression works both symbolically (SymPy) and numerically (NumPy)
- **Trainable Parameters**: Mathematical expressions that learn from data
- **Micro Autograd Engine**: Built-in automatic differentiation
- **Educational Model Builder**: Beginner-friendly model construction with built-in explanations
- **Educational Trainer**: Training with real-time tips, warnings, and debugging advice
- **Framework Bridges**: Convert to/from PyTorch, TensorFlow, JAX
- **Interactive Tutorials**: Step-by-step guided tutorials for learning ML/DL concepts
- **Composable**: Build complex expressions by combining simple ones
- **Searchable**: Find the right expression with built-in search
- **Visualization**: Plot expressions and training curves
- **Lightweight**: Core dependencies are just NumPy, SymPy, Matplotlib, SciPy

## 📦 Installation

### Basic

```bash
pip install neurogebra
```

### With optional extras

```bash
pip install neurogebra[viz]        # Interactive visualization (Plotly)
pip install neurogebra[fast]       # Performance optimizations (Numba)
pip install neurogebra[frameworks] # PyTorch + TensorFlow bridges
pip install neurogebra[dev]        # Development tools
pip install neurogebra[all]        # Everything
```

## 🧪 Usage Examples

### Working with Activations

```python
from neurogebra import MathForge

forge = MathForge()

# Get activation functions
sigmoid = forge.get("sigmoid")
tanh = forge.get("tanh")
swish = forge.get("swish")
gelu = forge.get("gelu")

# Evaluate
print(sigmoid.eval(x=0))   # 0.5
print(tanh.eval(x=0))      # 0.0

# Compute gradients symbolically
sigmoid_grad = sigmoid.gradient("x")
print(sigmoid_grad.formula)

# List all available activations
print(forge.list_all(category="activation"))
```

### Composing Expressions

```python
from neurogebra import MathForge

forge = MathForge()

# Compose loss functions
hybrid_loss = forge.compose("mse + 0.1*mae")

# Manual composition
f = forge.get("sigmoid")
g = forge.get("linear")
composed = f.compose(g)

# Arithmetic
mse = forge.get("mse")
mae = forge.get("mae")
custom_loss = 0.7 * mse + 0.3 * mae
```

### Training Expressions

```python
import numpy as np
from neurogebra import Expression
from neurogebra.core.trainer import Trainer

# Create a trainable expression
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
trainer = Trainer(expr, learning_rate=0.01)
history = trainer.fit(X, y, epochs=200, verbose=True)

print(f"Learned: m={expr.params['m']:.2f}, b={expr.params['b']:.2f}")
```

### Autograd Engine

```python
from neurogebra.core.autograd import Value

# Build computation graph
x = Value(2.0)
w = Value(-3.0)
b = Value(1.0)

# Forward pass
y = w * x + b
z = y.relu()

# Backward pass
z.backward()

print(f"dy/dw = {w.grad}")  # x = 2.0
print(f"dy/dx = {x.grad}")  # w = -3.0
```

### Search and Discovery

```python
from neurogebra import MathForge

forge = MathForge()

# Search by keyword
results = forge.search("classification")
print(results)

# List by category
print(forge.list_all(category="activation"))
print(forge.list_all(category="loss"))

# Compare expressions
print(forge.compare(["relu", "sigmoid", "tanh"]))
```

## 📚 Documentation

Full documentation is available at [https://neurogebra.readthedocs.io](https://neurogebra.readthedocs.io)

- [Getting Started](https://neurogebra.readthedocs.io/getting-started)
- [Tutorials](https://neurogebra.readthedocs.io/tutorials)
- [API Reference](https://neurogebra.readthedocs.io/api)

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Clone & setup
git clone https://github.com/fahiiim/NeuroGebra.git
cd NeuroGebra
python -m venv venv
venv\Scripts\activate  # or source venv/bin/activate on Unix
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Code quality
black src/ tests/
ruff check src/ tests/
```

## 📊 Expression Library

| Module | Count | Examples |
|--------|-------|---------|
| Activations | 15 | ReLU, Sigmoid, Tanh, Swish, GELU, Mish |
| Losses | 8 | MSE, MAE, Cross-Entropy, Huber, Hinge |
| Regularizers | 20 | L1, L2, Elastic Net, Dropout, SCAD, MCP |
| Algebra | 48 | Polynomials, Kernels, Distributions, Special Functions |
| Calculus | 48 | Derivatives, Integrals, Taylor Series, Transforms |
| Statistics | 35 | PDFs, CDFs, Bayesian, Information Theory, Regression |
| Linear Algebra | 24 | Norms, Distances, Matrix Ops, Attention |
| Optimization | 27 | SGD, Adam, AdamW, LR Schedules, Loss Landscapes |
| Metrics | 27 | F1, Precision, Recall, R², AIC, BIC, NDCG |
| Transforms | 33 | Normalization, Encoding, Initialization, Signal |
| **Total** | **285** | |

## 📄 License

MIT License - see [LICENSE](LICENSE)

## 🌟 Why Neurogebra?

| Feature | Neurogebra | NumPy | SymPy | Mathematica |
|---------|-----------|-------|-------|-------------|
| Symbolic Math | ✅ | ❌ | ✅ | ✅ |
| Numerical Eval | ✅ | ✅ | ⚠️ | ✅ |
| Autograd | ✅ | ❌ | ❌ | ⚠️ |
| Pre-built ML Expressions | ✅ (285) | ❌ | ❌ | ❌ |
| Educational Metadata | ✅ | ❌ | ❌ | ❌ |
| Trainable Formulas | ✅ | ❌ | ❌ | ❌ |
| Free & Open Source | ✅ | ✅ | ✅ | ❌ |
| Python Native | ✅ | ✅ | ✅ | ❌ |

## 👤 Author

**Fahim Sarker** — [@fahiiim](https://github.com/fahiiim)

---

*Made with ❤️ for the AI & Mathematics community*
