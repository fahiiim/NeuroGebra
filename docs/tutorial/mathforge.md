# MathForge — The Core

`MathForge` is the **central hub** of Neurogebra. It's the first thing you create and the main way you interact with the library.

---

## What is MathForge?

MathForge is a **repository manager** that gives you access to 50+ pre-built mathematical expressions used in Machine Learning.

Think of it as a **library catalog** — you ask for a book (expression) by name, and it gives you a full object you can work with.

---

## Creating MathForge

```python
from neurogebra import MathForge

forge = MathForge()
```

When you create a `MathForge`, it automatically loads all expressions from these categories:

- Activations (ReLU, Sigmoid, Tanh, etc.)
- Losses (MSE, MAE, Cross-Entropy, etc.)
- Regularizers (L1, L2, Elastic Net)
- Algebra (Polynomial, Quadratic, etc.)
- Calculus (Derivative-related expressions)
- Statistics (Mean, Variance, etc.)
- Linear Algebra (Dot product, Norms, etc.)
- Metrics (Accuracy, Precision, etc.)
- Optimization (Gradient descent steps)
- Transforms (Normalization, Standardization)

---

## Getting Expressions

### Basic Get

```python
# Get by name
relu = forge.get("relu")
sigmoid = forge.get("sigmoid")
mse = forge.get("mse")
```

### Get with Custom Parameters

```python
# Leaky ReLU has an 'alpha' parameter (default 0.01)
leaky = forge.get("leaky_relu")
print(leaky.params)  # {'alpha': 0.01}

# Override it
leaky_custom = forge.get("leaky_relu", params={"alpha": 0.2})
print(leaky_custom.params)  # {'alpha': 0.2}
```

### Get as Trainable

```python
# Make all parameters trainable
trainable_expr = forge.get("leaky_relu", trainable=True)
# Now 'alpha' can be learned from data
```

---

## Listing Expressions

### List Everything

```python
all_exprs = forge.list_all()
print(f"Total expressions: {len(all_exprs)}")
print(all_exprs)
```

### List by Category

```python
# Activation functions
activations = forge.list_all(category="activation")
print("Activations:", activations)

# Loss functions
losses = forge.list_all(category="loss")
print("Losses:", losses)

# Regularizers
regularizers = forge.list_all(category="regularizer")
print("Regularizers:", regularizers)
```

---

## Searching

Don't know the exact name? Search for it:

```python
# Search by keyword
results = forge.search("smooth")
print(results)

results = forge.search("classification")
print(results)

results = forge.search("gradient")
print(results)
```

---

## Composing Expressions

Combine expressions using string notation:

```python
# Custom loss = MSE + 0.1 * MAE
hybrid_loss = forge.compose("mse + 0.1*mae")

# Evaluate the composed expression
result = hybrid_loss.eval(y_pred=3.0, y_true=5.0)
print(f"Hybrid loss: {result}")
```

---

## Registering Custom Expressions

You can add your own expressions to the forge:

```python
from neurogebra import Expression

# Create a custom expression
my_func = Expression(
    "my_activation",
    "x * tanh(x)",
    metadata={"category": "activation", "description": "x*tanh(x) activation"}
)

# Register it
forge.register("my_activation", my_func)

# Now you can get it like any built-in expression
retrieved = forge.get("my_activation")
print(retrieved.eval(x=2.0))
```

---

## Complete Example

```python
from neurogebra import MathForge

forge = MathForge()

# 1. Explore what's available
print("=== Available Categories ===")
for category in ["activation", "loss", "regularizer"]:
    exprs = forge.list_all(category=category)
    print(f"  {category}: {len(exprs)} expressions")

# 2. Get and explore an expression
print("\n=== Sigmoid ===")
sigmoid = forge.get("sigmoid")
print(f"  Formula: {sigmoid.symbolic_expr}")
print(f"  sigmoid(0) = {sigmoid.eval(x=0)}")
print(f"  sigmoid(5) = {sigmoid.eval(x=5):.4f}")

# 3. Compare activations
print("\n=== Activation Comparison at x=1.0 ===")
for name in ["relu", "sigmoid", "tanh", "swish", "gelu"]:
    expr = forge.get(name)
    val = expr.eval(x=1.0)
    print(f"  {name:>10}: {val:.4f}")

# 4. Compose a hybrid loss
print("\n=== Hybrid Loss ===")
loss = forge.compose("mse + 0.1*mae")
error = loss.eval(y_pred=2.5, y_true=3.0)
print(f"  Loss value: {error:.4f}")
```

---

## Quick Reference

| Method | Description | Example |
|--------|-------------|---------|
| `forge.get(name)` | Get expression by name | `forge.get("relu")` |
| `forge.get(name, params={})` | Get with custom params | `forge.get("leaky_relu", params={"alpha": 0.2})` |
| `forge.list_all()` | List all expressions | `forge.list_all()` |
| `forge.list_all(category=)` | List by category | `forge.list_all(category="activation")` |
| `forge.search(query)` | Search expressions | `forge.search("smooth")` |
| `forge.compose(str)` | Compose from string | `forge.compose("mse + 0.1*mae")` |
| `forge.register(name, expr)` | Add custom expression | `forge.register("my_fn", expr)` |

---

**Next:** [Expressions — Building Blocks →](expressions.md)
