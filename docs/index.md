# Welcome to Neurogebra

## Learn Machine Learning from Scratch — The Easy Way

---

**Neurogebra** is a Python framework built for **students and beginners** who want to learn Machine Learning and Deep Learning by actually understanding the math behind it — not just calling black-box functions.

!!! tip "Who is this for?"
    If you know **basic Python** and want to start your **ML/AI journey**, you're in the right place. No prior ML experience needed.

---

## What Makes Neurogebra Different?

| Feature | Traditional Frameworks (PyTorch, TF) | Neurogebra |
|---------|--------------------------------------|------------|
| Learning curve | Steep — many hidden abstractions | Gentle — every step is explained |
| Math visibility | Hidden inside C++ kernels | Symbolic — you SEE the formulas |
| Expressions | Tensors/Modules you don't understand | Mathematical expressions you can read |
| Explanations | Read research papers | Built-in `.explain()` on everything |
| Target audience | Production engineers | Students & learners |

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

This documentation is structured like a **step-by-step course**, starting from absolute basics:

### 📘 Getting Started
Install Neurogebra and write your first program in under 5 minutes.

### 🐍 Python for ML (Refresher)
A quick refresher on Python basics, NumPy, and data handling — the prerequisites for ML.

### 🧠 ML Fundamentals
What Machine Learning actually is, the types of ML, the standard workflow, and the math behind it all.

### 🔧 Neurogebra Tutorial
Step-by-step lessons on every Neurogebra feature — from expressions to training to model building.

### 🚀 Advanced Topics
Custom expressions, framework bridges (PyTorch/TF/JAX), visualization, and optimization.

### 🏗️ Full Projects (Neurogebra vs PyTorch)
3 complete ML/Deep Learning projects with **side-by-side code comparison** between Neurogebra and PyTorch, so you understand the value of this framework.

---

## Install Now

```bash
pip install neurogebra
```

**Ready?** Start with [Installation →](getting-started/installation.md)
