# Expressions — Building Blocks

The `Expression` class is the **fundamental unit** of Neurogebra. Everything in the library is built on expressions.

---

## What is an Expression?

An Expression is a **symbolic mathematical formula** that can:

- ✅ **Evaluate** — compute a numerical result
- ✅ **Differentiate** — compute gradients symbolically
- ✅ **Explain** — describe itself in plain English
- ✅ **Compose** — combine with other expressions
- ✅ **Train** — learn parameters from data
- ✅ **Visualize** — plot itself

---

## Creating an Expression

```python
from neurogebra import Expression

# Simple: name + formula
square = Expression("square", "x**2")

# With parameters
line = Expression("line", "m*x + b", params={"m": 2.0, "b": 1.0})

# With trainable parameters
trainable_line = Expression(
    "trainable_line",
    "m*x + b",
    params={"m": 0.0, "b": 0.0},
    trainable_params=["m", "b"]
)

# With metadata
custom = Expression(
    "my_function",
    "x * tanh(x)",
    metadata={
        "category": "activation",
        "description": "Custom activation function",
        "usage": "Experimental hidden layers"
    }
)
```

---

## Evaluating Expressions

### With Keyword Arguments

```python
expr = Expression("poly", "a*x**2 + b*x + c",
                  params={"a": 1.0, "b": -2.0, "c": 1.0})

# Parameters are substituted automatically
result = expr.eval(x=3)
print(result)  # 1*9 + (-2)*3 + 1 = 4.0
```

### With NumPy Arrays

```python
import numpy as np

square = Expression("square", "x**2")
x = np.array([1, 2, 3, 4, 5])
result = square.eval(x=x)
print(result)  # [ 1  4  9 16 25]
```

### Expression with Multiple Variables

```python
loss = Expression("mse", "(y_pred - y_true)**2")
result = loss.eval(y_pred=3.0, y_true=5.0)
print(result)  # 4.0
```

---

## Expression Properties

```python
from neurogebra import MathForge

forge = MathForge()
sigmoid = forge.get("sigmoid")

# Name
print(sigmoid.name)             # "sigmoid"

# Symbolic formula (SymPy expression)
print(sigmoid.symbolic_expr)    # 1/(1 + exp(-x))

# Variables (free symbols)
print(sigmoid.variables)        # [x]

# Parameters
print(sigmoid.params)           # {}

# Metadata
print(sigmoid.metadata)
# {'category': 'activation', 'description': '...', ...}
```

---

## Explaining Expressions

Every expression can explain itself:

```python
relu = forge.get("relu")
print(relu.explain())

sigmoid = forge.get("sigmoid")
print(sigmoid.explain())

# Metadata gives you detailed info
print(sigmoid.metadata["description"])
print(sigmoid.metadata["usage"])
print(sigmoid.metadata["pros"])
print(sigmoid.metadata["cons"])
```

---

## Expression Arithmetic

Expressions support standard math operations:

```python
from neurogebra import Expression

a = Expression("a", "x**2")
b = Expression("b", "x + 1")

# Addition
c = a + b           # x² + x + 1

# Subtraction  
d = a - b           # x² - x - 1

# Multiplication
e = a * b           # x² * (x + 1) = x³ + x²

# Scalar multiplication
f = 2 * a           # 2x²
g = a * 0.5         # 0.5x²
```

---

## Cloning Expressions

Create independent copies:

```python
original = forge.get("relu")
clone = original.clone()

# Modifying clone doesn't affect original
clone.params["new_param"] = 42
print("new_param" in original.params)  # False
```

---

## Calling Expressions

Expressions are callable (you can use them like functions):

```python
relu = forge.get("relu")

# These are equivalent:
result1 = relu.eval(x=5)
result2 = relu(5)
print(result1, result2)  # 5, 5
```

---

## Complete Example

```python
from neurogebra import Expression, MathForge
import numpy as np

# === Create Custom Expression ===
gaussian = Expression(
    "gaussian",
    "exp(-x**2 / (2 * sigma**2))",
    params={"sigma": 1.0},
    metadata={
        "category": "statistics",
        "description": "Gaussian (bell curve) function"
    }
)

# === Evaluate ===
x = np.linspace(-3, 3, 7)
y = gaussian.eval(x=x)
print("Gaussian values:")
for xi, yi in zip(x, y):
    print(f"  x={xi:+.1f}  →  f(x)={yi:.4f}")

# === Gradient ===
grad = gaussian.gradient("x")
print(f"\nGradient formula: {grad.symbolic_expr}")

# === Change parameter ===
gaussian.params["sigma"] = 0.5  # Narrower bell curve
y_narrow = gaussian.eval(x=x)
print("\nNarrower Gaussian:")
for xi, yi in zip(x, y_narrow):
    print(f"  x={xi:+.1f}  →  f(x)={yi:.4f}")
```

---

## Quick Reference

| Property/Method | Description | Example |
|----------------|-------------|---------|
| `.name` | Expression name | `"relu"` |
| `.symbolic_expr` | SymPy formula | `Max(0, x)` |
| `.variables` | Free variables | `[x]` |
| `.params` | Parameter dict | `{"alpha": 0.01}` |
| `.metadata` | Extra info | `{"category": "activation"}` |
| `.eval(**kwargs)` | Evaluate numerically | `.eval(x=5)` |
| `.gradient(var)` | Symbolic derivative | `.gradient("x")` |
| `.compose(other)` | Compose: self(other) | `.compose(linear)` |
| `.clone()` | Deep copy | `.clone()` |
| `.explain()` | Plain English explanation | `.explain()` |
| `.visualize()` | Plot the expression | `.visualize()` |

---

**Next:** [Activation Functions →](activations.md)
