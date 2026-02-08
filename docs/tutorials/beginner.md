# Beginner Tutorial

Welcome to Neurogebra! This tutorial covers the fundamental concepts.

## What is Neurogebra?

Neurogebra is a mathematics library designed for AI developers. It lets you work with mathematical expressions that are:

- **Symbolic** - See and manipulate formulas
- **Numerical** - Evaluate efficiently with NumPy
- **Trainable** - Learn parameters from data
- **Educational** - Understand what you're using

## Step 1: Create a MathForge

MathForge is your main entry point:

```python
from neurogebra import MathForge

forge = MathForge()
```

## Step 2: Get an Expression

```python
# Get ReLU activation
relu = forge.get("relu")

# See what it is
print(relu)           # Max(0, x)
print(relu.formula)   # LaTeX
print(relu.explain())  # Full explanation
```

## Step 3: Evaluate

```python
import numpy as np

# Single value
result = relu.eval(x=5)    # 5
result = relu.eval(x=-3)   # 0

# Array
result = relu.eval(x=np.array([-2, -1, 0, 1, 2]))
# [0, 0, 0, 1, 2]
```

## Step 4: Compute Gradients

```python
# Symbolic gradient
relu_grad = relu.gradient("x")
print(relu_grad)  # Derivative expression

# Evaluate gradient
grad_value = relu_grad.eval(x=2)
```

## Step 5: Explore

```python
# List all available expressions
all_exprs = forge.list_all()

# Category-wise
activations = forge.list_all(category="activation")
losses = forge.list_all(category="loss")

# Search
results = forge.search("smooth")
```

## What's Next?

- Try different activations: `sigmoid`, `tanh`, `swish`, `gelu`
- Look at loss functions: `mse`, `mae`, `huber`
- Move to the [Intermediate Tutorial](intermediate.md)
