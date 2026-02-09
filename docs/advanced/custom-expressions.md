# Custom Expressions

Learn how to create your own mathematical expressions and register them for reuse.

---

## Why Custom Expressions?

While Neurogebra ships with 50+ built-in expressions, you'll often want:

- Custom activation functions for research
- Specialized loss functions for your task
- Domain-specific mathematical formulas
- Experimental combinations

---

## Creating a Custom Expression

### Basic Custom Expression

```python
from neurogebra import Expression

# Simple: just name and formula
my_func = Expression("my_square", "x**2 + 1")

print(my_func.eval(x=3))          # 10.0
print(my_func.gradient("x").eval(x=3))  # 6.0
```

### With Parameters

```python
# Parametric expression
gaussian = Expression(
    "gaussian",
    "A * exp(-(x - mu)**2 / (2 * sigma**2))",
    params={"A": 1.0, "mu": 0.0, "sigma": 1.0}
)

print(gaussian.eval(x=0))    # 1.0 (peak at mu=0)
print(gaussian.eval(x=1))    # ≈ 0.607
print(gaussian.eval(x=-1))   # ≈ 0.607
```

### With Trainable Parameters

```python
# Parameters that can be learned from data
learnable = Expression(
    "adaptive_relu",
    "Max(alpha * x, x)",
    params={"alpha": 0.01},
    trainable_params=["alpha"],  # This will be optimized
    metadata={
        "category": "activation",
        "description": "PReLU - Parametric ReLU with learnable slope"
    }
)
```

### With Full Metadata

```python
my_activation = Expression(
    "mish",
    "x * tanh(log(1 + exp(x)))",
    metadata={
        "category": "activation",
        "description": "Mish activation function",
        "usage": "Modern deep networks, alternative to Swish",
        "pros": ["Self-regularizing", "Smooth", "Non-monotonic"],
        "cons": ["Computationally expensive"],
        "paper": "Mish: A Self Regularized Non-Monotonic Activation Function"
    }
)

print(my_activation.metadata["description"])
print(my_activation.metadata["pros"])
```

---

## Registering Custom Expressions

Once created, register with MathForge for easy access:

```python
from neurogebra import MathForge, Expression

forge = MathForge()

# Create
mish = Expression(
    "mish",
    "x * tanh(log(1 + exp(x)))",
    metadata={"category": "activation"}
)

# Register
forge.register("mish", mish)

# Now use it like any built-in
retrieved = forge.get("mish")
print(retrieved.eval(x=1.0))

# It appears in searches too
results = forge.search("mish")
print(results)  # ['mish']
```

---

## Custom Activation Functions

### ELU (Exponential Linear Unit)

```python
elu = Expression(
    "elu",
    "Piecewise((x, x > 0), (alpha*(exp(x) - 1), True))",
    params={"alpha": 1.0},
    metadata={
        "category": "activation",
        "description": "ELU - smooth alternative to ReLU for negative values"
    }
)
```

### SELU (Scaled ELU)

```python
selu = Expression(
    "selu",
    "scale * Piecewise((x, x > 0), (alpha*(exp(x) - 1), True))",
    params={"scale": 1.0507, "alpha": 1.6733},
    metadata={
        "category": "activation",
        "description": "Self-normalizing activation"
    }
)
```

### Hard Sigmoid

```python
hard_sigmoid = Expression(
    "hard_sigmoid",
    "Max(0, Min(1, 0.2*x + 0.5))",
    metadata={
        "category": "activation",
        "description": "Computationally cheap approximation of sigmoid"
    }
)
```

---

## Custom Loss Functions

### Smooth L1 Loss

```python
smooth_l1 = Expression(
    "smooth_l1",
    "Piecewise((0.5 * (y_pred - y_true)**2, Abs(y_pred - y_true) < 1), (Abs(y_pred - y_true) - 0.5, True))",
    metadata={
        "category": "loss",
        "description": "Smooth L1 loss - used in object detection"
    }
)
```

### Weighted MSE

```python
weighted_mse = Expression(
    "weighted_mse",
    "weight * (y_pred - y_true)**2",
    params={"weight": 1.0},
    metadata={
        "category": "loss",
        "description": "MSE with sample weight"
    }
)
```

---

## Testing Custom Expressions

Always test your expressions thoroughly:

```python
import numpy as np
from neurogebra import Expression

# Create expression
my_func = Expression("test", "x**3 - 3*x + 2")

# Test basic evaluation
assert my_func.eval(x=0) == 2.0
assert my_func.eval(x=1) == 0.0

# Test gradient
grad = my_func.gradient("x")
assert grad.eval(x=0) == -3.0    # f'(0) = -3
assert grad.eval(x=1) == 0.0     # f'(1) = 0

# Test array evaluation
x = np.array([-2, -1, 0, 1, 2])
result = my_func.eval(x=x)
expected = x**3 - 3*x + 2
np.testing.assert_array_almost_equal(result, expected)

print("All tests passed! ✓")
```

---

**Next:** [Framework Bridges →](framework-bridges.md)
