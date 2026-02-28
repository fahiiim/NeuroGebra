# Welcome to Neurogebra

## The Executable Mathematical Formula Companion for AI and Data Science

**v2.5.3** | 285 Symbolic Expressions | 100+ Datasets | Training Observatory Pro

---

**Neurogebra** is a unified Python library that bridges **symbolic mathematics**, **numerical computation**, and **machine learning**. It provides 285 pre-built, tested, and documented mathematical expressions, 100+ curated datasets, and a training system with full mathematical transparency -- designed for students, researchers, and engineers alike.

!!! tip "Who is this for?"
    - **Students** learning ML/AI who want to see the math behind every operation
    - **Researchers** who need reproducibility, rapid formula prototyping, and transparent diagnostics
    - **Engineers** who want a verified formula library with production-ready logging

---

## What Makes Neurogebra Different?

| Feature | Traditional Frameworks (PyTorch, TF) | Neurogebra |
|---------|--------------------------------------|------------|
| Learning curve | Steep -- many hidden abstractions | Gentle -- every step is explained |
| Math visibility | Hidden inside C++ kernels | Symbolic -- you SEE the formulas |
| Expressions | Tensors/Modules you don't understand | Mathematical expressions you can read |
| Explanations | Read research papers | Built-in `.explain()` on everything |
| Training diagnostics | Basic loss curves | Full math transparency with Observatory Pro |
| Reproducibility | Manual tracking | Automatic training fingerprinting |
| Formula library | Build your own | 285 verified, searchable, composable formulas |
| Target audience | Production engineers | Students, researchers, and engineers |

---

## Quick Example

```python
from neurogebra import MathForge

# Create the main interface
forge = MathForge()

# Get the ReLU activation function
relu = forge.get("relu")

# See what it actually IS
print(relu.explain())
# Output: "ReLU (Rectified Linear Unit) outputs x if x > 0, else 0"

# See the formula
print(relu.formula)
# Output: Max(0, x)

# Evaluate it
print(relu.eval(x=5))    # 5
print(relu.eval(x=-3))   # 0

# Get its derivative (gradient)
relu_grad = relu.gradient("x")
print(relu_grad)
```

!!! success "See how readable that is?"
    Every expression tells you what it is, how it works, and why it matters. No magic. No black boxes.

---

## What You'll Learn in This Documentation

This documentation is structured as a **progressive learning path**, starting from absolute basics:

### Getting Started
Install Neurogebra and write your first program in under 5 minutes.

### Python for ML (Refresher)
A quick refresher on Python basics, NumPy, and data handling -- the prerequisites for ML.

### ML Fundamentals
What Machine Learning actually is, the types of ML, the standard workflow, and the math behind it all.

### Neurogebra Tutorial
Step-by-step lessons on every feature -- from expressions to training to model building.

### Training Observatory and Observatory Pro
Real-time training diagnostics with adaptive logging, automated health warnings, epoch summaries, visual dashboards, and training fingerprinting.

### Advanced Topics
Custom expressions, framework bridges (PyTorch / TF / JAX), visualization, and optimization.

### Full Projects (Neurogebra vs PyTorch)
3 complete ML/Deep Learning projects with **side-by-side code comparison** between Neurogebra and PyTorch.

---

## Install Now

```bash
pip install neurogebra
```

**Ready?** Start with [Installation →](getting-started/installation.md)
