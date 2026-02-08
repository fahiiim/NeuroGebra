# Getting Started with Neurogebra

## Installation

```bash
pip install neurogebra
```

For optional features:
```bash
pip install neurogebra[viz]        # Visualization tools
pip install neurogebra[fast]       # Performance optimizations
pip install neurogebra[frameworks] # PyTorch, TensorFlow support
pip install neurogebra[all]        # Everything
```

## Your First Expression

```python
from neurogebra import MathForge

# Create forge instance
forge = MathForge()

# Get an activation function
relu = forge.get("relu")

# Evaluate it
result = relu.eval(x=5)
print(result)  # 5

result = relu.eval(x=-3)
print(result)  # 0
```

## Understanding Expressions

```python
# Get explanation
print(relu.explain())

# See the formula (LaTeX)
print(relu.formula)

# Get gradient
relu_grad = relu.gradient("x")
print(relu_grad)
```

## Exploring Available Expressions

```python
# List all expressions
print(forge.list_all())

# List by category
print(forge.list_all(category="activation"))
print(forge.list_all(category="loss"))

# Search
results = forge.search("classification")
print(results)
```

## Composing Expressions

```python
# Get multiple expressions
mse = forge.get("mse")
mae = forge.get("mae")

# Arithmetic composition
hybrid_loss = 0.7 * mse + 0.3 * mae

# String-based composition
custom_loss = forge.compose("mse + 0.1*mae")
```

## Training Expressions

```python
import numpy as np
from neurogebra import Expression
from neurogebra.core.trainer import Trainer

# Create trainable expression
expr = Expression(
    "my_line",
    "m*x + b",
    params={"m": 0.0, "b": 0.0},
    trainable_params=["m", "b"]
)

# Generate synthetic data
X = np.linspace(0, 5, 50)
y = 2 * X + 1

# Train
trainer = Trainer(expr, learning_rate=0.01)
history = trainer.fit(X, y, epochs=100)

print(f"Learned: m={expr.params['m']:.2f}, b={expr.params['b']:.2f}")
```

## Next Steps

- [Beginner Tutorial](tutorials/beginner.md) - Learn the fundamentals
- [Intermediate Tutorial](tutorials/intermediate.md) - Advanced features
- [Advanced Tutorial](tutorials/advanced.md) - Expert-level usage
