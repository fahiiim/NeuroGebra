# How Neurogebra Works

This page explains the **architecture** of Neurogebra — how all the pieces fit together.

---

## The Architecture

Neurogebra is built in layers, from low-level math to high-level model building:

```
┌─────────────────────────────────────────┐
│          ModelBuilder / NeuroCraft       │  ← High-level: Build models
├─────────────────────────────────────────┤
│              Trainer                     │  ← Train expressions on data
├─────────────────────────────────────────┤
│              MathForge                   │  ← Access expressions
├─────────────────────────────────────────┤
│         Expression + Autograd            │  ← Core: Math + Gradients
├─────────────────────────────────────────┤
│    Repository (Activations, Losses...)   │  ← Pre-built formulas
├─────────────────────────────────────────┤
│          SymPy + NumPy                   │  ← Foundation: Math engines
└─────────────────────────────────────────┘
```

---

## Core Components

### 1. Expression — The Building Block

Everything in Neurogebra is an **Expression**. An Expression is a mathematical formula that can:

- **Evaluate** → give you a number
- **Differentiate** → compute its gradient
- **Explain** → describe itself in plain English
- **Compose** → combine with other expressions
- **Train** → learn parameter values from data

```python
from neurogebra import Expression

# A simple expression: y = mx + b
line = Expression(
    name="line",
    symbolic_expr="m*x + b",
    params={"m": 2.0, "b": 1.0}
)

print(line.eval(x=3))  # 2*3 + 1 = 7.0
```

### 2. MathForge — The Toolbox

MathForge loads all pre-built expressions from the **Repository** and gives you a clean interface:

```python
from neurogebra import MathForge

forge = MathForge()

# Access 50+ expressions by name
relu = forge.get("relu")
mse = forge.get("mse")
```

### 3. Repository — The Expression Library

The repository contains pre-built expressions organized by category:

| Category | Examples | Used For |
|----------|----------|----------|
| Activations | ReLU, Sigmoid, Tanh, GELU, Swish | Adding non-linearity to neural networks |
| Losses | MSE, MAE, Cross-Entropy, Huber | Measuring how wrong predictions are |
| Regularizers | L1, L2, Elastic Net | Preventing overfitting |
| Algebra | Polynomial, Quadratic | General mathematical operations |
| Calculus | Derivatives, Integrals | Mathematical analysis |
| Statistics | Mean, Variance, Standard Deviation | Data analysis |
| Linear Algebra | Dot product, Norms | Vector/matrix operations |
| Metrics | Accuracy, Precision, Recall | Evaluating model performance |
| Optimization | Gradient Descent step | Training algorithms |
| Transforms | Normalization, Standardization | Data preprocessing |

### 4. Autograd Engine — The Gradient Machine

The autograd engine tracks computations and computes gradients automatically:

```python
from neurogebra.core.autograd import Value

x = Value(2.0)
y = Value(3.0)

# Forward: compute result
z = x * y + x ** 2  # z = 2*3 + 2² = 10

# Backward: compute gradients
z.backward()

print(x.grad)  # dz/dx = y + 2x = 3 + 4 = 7.0
print(y.grad)  # dz/dy = x = 2.0
```

### 5. Trainer — The Learning Engine

The Trainer fits expression parameters to data:

```python
from neurogebra import Expression
from neurogebra.core.trainer import Trainer
import numpy as np

# Expression with unknown parameters
expr = Expression("line", "m*x + b",
                  params={"m": 0.0, "b": 0.0},
                  trainable_params=["m", "b"])

# Data: y = 2x + 1
X = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([3, 5, 7, 9, 11], dtype=float)

# Train to find m and b
trainer = Trainer(expr, learning_rate=0.01)
trainer.fit(X, y, epochs=200)

print(f"Learned: m={expr.params['m']:.2f}, b={expr.params['b']:.2f}")
# Should be close to m=2.0, b=1.0
```

### 6. ModelBuilder — The Neural Network Constructor

ModelBuilder lets you build neural networks with an educational interface:

```python
from neurogebra import ModelBuilder

builder = ModelBuilder()

model = builder.Sequential([
    builder.Dense(128, activation="relu"),
    builder.Dropout(0.2),
    builder.Dense(64, activation="relu"),
    builder.Dense(10, activation="softmax")
])

model.summary()            # See architecture
model.explain_architecture()  # Get explanations
```

---

## How It All Connects

Here's the typical workflow:

```
1. You create a MathForge           →  forge = MathForge()
2. You get expressions               →  relu = forge.get("relu")
3. You explore them                   →  relu.explain(), relu.eval(x=5)
4. You compose complex expressions    →  loss = forge.compose("mse + 0.1*mae")
5. You create trainable expressions   →  expr = Expression("f", "a*x+b", ...)
6. You train on data                  →  trainer.fit(X, y, epochs=100)
7. You build full models              →  model = builder.Sequential([...])
```

---

## Symbolic vs Numerical

Neurogebra uses **SymPy** for symbolic math and **NumPy** for numerical computation:

```python
relu = forge.get("relu")

# SYMBOLIC — see the formula
print(relu.symbolic_expr)  # Max(0, x)

# NUMERICAL — get a number
print(relu.eval(x=5))     # 5
```

**Why both?**

- **Symbolic**: You can see, manipulate, differentiate, and explain formulas
- **Numerical**: You can evaluate efficiently on real data with NumPy arrays

---

## Summary

| Component | Class | Purpose |
|-----------|-------|---------|
| Expression | `Expression` | A math formula you can evaluate, differentiate, train |
| MathForge | `MathForge` | Toolbox to access all pre-built expressions |
| Value | `Value` | Scalar with automatic gradient tracking |
| Tensor | `Tensor` | Array with automatic gradient tracking |
| Trainer | `Trainer` | Fits expression parameters to data |
| ModelBuilder | `ModelBuilder` | Builds neural network architectures |
| NeuroCraft | `NeuroCraft` | Enhanced interface with educational features |

---

**Next:** [Python for ML →](../python-refresher/basics.md)
