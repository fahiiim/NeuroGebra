# NeuroCraft Interface

`NeuroCraft` is Neurogebra's **enhanced educational interface** — like MathForge but with built-in learning tools, tutorials, and smarter error handling.

---

## NeuroCraft vs MathForge

| Feature | MathForge | NeuroCraft |
|---------|-----------|------------|
| Get expressions | ✅ | ✅ |
| Search | ✅ | ✅ Smart with suggestions |
| Educational mode | ❌ | ✅ Built-in explanations |
| Tutorials | ❌ | ✅ Interactive tutorials |
| "Did you mean?" | ❌ | ✅ Typo correction |
| Compare expressions | ❌ | ✅ Side-by-side comparison |
| Quick access | ❌ | ✅ `quick_activation()`, `quick_loss()` |

---

## Getting Started

```python
from neurogebra import NeuroCraft

# Educational mode ON (default) — shows tips and explanations
craft = NeuroCraft(educational_mode=True)
# Output:
# 🎓 Welcome to Neurogebra!
#    Type craft.tutorial() to start learning
#    Type craft.search('activation') to explore

# Quiet mode — for experienced users
craft = NeuroCraft(educational_mode=False)
```

---

## Getting Expressions

```python
# Standard get
relu = craft.get("relu")

# With instant explanation
sigmoid = craft.get("sigmoid", explain=True)
# Automatically prints explanation when educational_mode=True

# With parameter override
leaky = craft.get("leaky_relu", params={"alpha": 0.2})
```

### Smart Error Handling

```python
# If you mistype:
try:
    craft.get("rellu")  # Typo!
except KeyError as e:
    print(e)
# Expression 'rellu' not found.
#    Did you mean: relu, leaky_relu, gelu?
#    Use craft.search('rellu') to find related expressions.
```

---

## Searching with NeuroCraft

```python
# Search with detailed output
results = craft.search("activation")
# Prints a formatted list with descriptions

# Search by category
results = craft.search("loss", category="loss")

# Search specific terms
results = craft.search("smooth")
results = craft.search("classification")
```

---

## Quick Access Methods

```python
# Quick activation — get + explain in one call
relu = craft.quick_activation("relu")

# Quick loss
mse = craft.quick_loss("mse")

# These automatically show explanations in educational mode
```

---

## Comparing Expressions

```python
# Visual comparison of activations
craft.compare(
    ["relu", "sigmoid", "tanh", "swish"],
    metric="behavior"
)

# Compare gradients
craft.compare(
    ["relu", "sigmoid", "tanh"],
    metric="gradient"
)
```

---

## Interactive Tutorials

```python
# Show tutorial menu
craft.tutorial()

# Start a specific tutorial
craft.tutorial("basics")
craft.tutorial("activations")
craft.tutorial("training")
craft.tutorial("first_model")
```

---

## Listing All Expressions

```python
# All expressions
all_exprs = craft.list_all()
print(f"Total: {len(all_exprs)}")

# By category
print("Activations:", craft.list_all(category="activation"))
print("Losses:", craft.list_all(category="loss"))
print("Regularizers:", craft.list_all(category="regularizer"))
```

---

## Composing with NeuroCraft

```python
# String composition
hybrid_loss = craft.compose("mse + 0.1*mae")

# Evaluate
result = hybrid_loss.eval(y_pred=3.0, y_true=5.0)
print(f"Loss: {result}")
```

---

## Registering Custom Expressions

```python
from neurogebra import Expression

# Create your own
my_activation = Expression(
    "xtanh",
    "x * tanh(x)",
    metadata={
        "category": "activation",
        "description": "x*tanh(x) — smooth activation"
    }
)

# Register it
craft.register("xtanh", my_activation)

# Now available everywhere
retrieved = craft.get("xtanh")
print(retrieved.eval(x=2.0))
```

---

## When to Use NeuroCraft vs MathForge

| Scenario | Use |
|----------|-----|
| Learning/exploring | **NeuroCraft** with `educational_mode=True` |
| Production code | **MathForge** (lighter, no prints) |
| Interactive notebooks | **NeuroCraft** |
| Scripts and pipelines | **MathForge** |
| Teaching others | **NeuroCraft** |

---

**Next:** [Datasets →](datasets.md)
