# Neurogebra

**Neural-powered mathematics for AI developers**

## Overview

Neurogebra is a unified Python library that bridges symbolic mathematics, numerical computation, and deep learning. It provides pre-built mathematical expressions that are evaluable, differentiable, composable, and trainable.

## Quick Start

```bash
pip install neurogebra
```

```python
from neurogebra import MathForge

forge = MathForge()

# Get activation functions
relu = forge.get("relu")
sigmoid = forge.get("sigmoid")

# Evaluate
relu.eval(x=5)      # 5
sigmoid.eval(x=0)   # 0.5

# Explain
relu.explain()

# Compose
custom_loss = forge.compose("mse + 0.1*mae")
```

## Features

- 50+ pre-built mathematical expressions
- Symbolic and numerical evaluation
- Automatic differentiation
- Trainable parameters
- Expression composition
- Framework bridges (PyTorch, TensorFlow, JAX)
- Educational explanations
- Visualization tools

## Navigation

- [Getting Started](getting-started.md) - Installation and first steps
- [Tutorials](tutorials/beginner.md) - Step-by-step guides
- [API Reference](api/reference.md) - Complete API documentation
- [Examples](examples/custom_activation.md) - Practical examples
