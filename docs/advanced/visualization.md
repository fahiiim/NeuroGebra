# Visualization

Neurogebra includes built-in tools to **visualize** expressions, gradients, and training progress.

---

## Setup

```bash
pip install neurogebra[viz]
# or
pip install matplotlib
```

---

## Plotting Expressions

### Single Expression

```python
from neurogebra import MathForge
from neurogebra.viz.plotting import plot_expression

forge = MathForge()
sigmoid = forge.get("sigmoid")

# Plot from x=-5 to x=5
plot_expression(sigmoid, x_range=(-5, 5), title="Sigmoid Function")
```

### Multiple Expressions Side by Side

```python
from neurogebra.viz.plotting import plot_comparison

forge = MathForge()

activations = {
    "ReLU": forge.get("relu"),
    "Sigmoid": forge.get("sigmoid"),
    "Tanh": forge.get("tanh"),
    "Swish": forge.get("swish")
}

plot_comparison(
    activations,
    x_range=(-5, 5),
    title="Activation Function Comparison"
)
```

---

## Plotting Gradients

Visualize how gradients behave — essential for understanding vanishing/exploding gradients:

```python
from neurogebra.viz.plotting import plot_gradient

forge = MathForge()
sigmoid = forge.get("sigmoid")

# Plot function and its gradient together
plot_gradient(
    sigmoid,
    x_range=(-5, 5),
    title="Sigmoid and Its Gradient"
)
```

---

## Training History

After training, visualize how the loss decreased:

```python
from neurogebra import MathForge, Expression
from neurogebra.core.trainer import Trainer
from neurogebra.viz.plotting import plot_training_history
import numpy as np

# Setup
forge = MathForge()
model = Expression("linear", "w * x + b", params={"w": 0.0, "b": 0.0}, trainable_params=["w", "b"])

# Generate data
X = np.linspace(0, 10, 50)
y = 2.5 * X + 1.0 + np.random.normal(0, 0.5, 50)

# Train
loss_fn = forge.get("mse")
trainer = Trainer(model, loss_fn, optimizer="adam", lr=0.01)
history = trainer.fit(X, y, epochs=100)

# Plot the training history
plot_training_history(history, title="Training Progress")
```

---

## Manual Visualization with Matplotlib

You can also create custom plots directly:

### Expression Evaluation Plot

```python
import numpy as np
import matplotlib.pyplot as plt
from neurogebra import MathForge

forge = MathForge()
relu = forge.get("relu")
sigmoid = forge.get("sigmoid")

x = np.linspace(-5, 5, 200)

plt.figure(figsize=(10, 6))
plt.plot(x, relu.eval(x=x), label="ReLU", linewidth=2)
plt.plot(x, sigmoid.eval(x=x), label="Sigmoid", linewidth=2)
plt.axhline(y=0, color='k', linewidth=0.5)
plt.axvline(x=0, color='k', linewidth=0.5)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Activation Functions")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### Gradient Comparison Plot

```python
import numpy as np
import matplotlib.pyplot as plt
from neurogebra import MathForge

forge = MathForge()

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
x = np.linspace(-5, 5, 200)

for ax, name in zip(axes.flat, ["relu", "sigmoid", "tanh", "swish"]):
    expr = forge.get(name)
    grad = expr.gradient("x")
    
    ax.plot(x, expr.eval(x=x), label=f"{name}(x)", linewidth=2)
    ax.plot(x, grad.eval(x=x), label=f"{name}'(x)", linewidth=2, linestyle="--")
    ax.set_title(name.upper())
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)

plt.tight_layout()
plt.show()
```

### Training Loss Curve

```python
import matplotlib.pyplot as plt

# After training, plot manually:
# history = trainer.fit(X, y, epochs=100)

plt.figure(figsize=(8, 5))
plt.plot(history["loss"], linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Time")
plt.grid(True, alpha=0.3)
plt.yscale("log")  # Log scale to see details
plt.show()
```

---

## Interactive Visualization

For Jupyter notebooks:

```python
from neurogebra.viz.interactive import interactive_plot

forge = MathForge()
sigmoid = forge.get("sigmoid")

# Interactive slider for parameters
interactive_plot(sigmoid, x_range=(-5, 5))
```

---

## Visualization Best Practices

| Tip | Why |
|-----|-----|
| Always label axes | Readability |
| Use `grid(True, alpha=0.3)` | Easier to read values |
| Compare related functions together | Better understanding |
| Plot gradients with functions | See the relationship |
| Use log scale for loss curves | See small improvements |
| Save with `plt.savefig("plot.png", dpi=150)` | High quality output |

---

**Next:** [Regularization →](regularization.md)
