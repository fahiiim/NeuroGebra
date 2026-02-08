# Training Expressions

Learn how to train mathematical expressions to fit data.

## Basic Training

```python
import numpy as np
from neurogebra import Expression
from neurogebra.core.trainer import Trainer

# Create trainable linear expression
expr = Expression(
    "line",
    "m*x + b",
    params={"m": 0.0, "b": 0.0},
    trainable_params=["m", "b"]
)

# Generate data
X = np.linspace(0, 10, 100)
y = 2.5 * X + 3.0 + np.random.normal(0, 0.5, 100)

# Train
trainer = Trainer(expr, learning_rate=0.001)
history = trainer.fit(X, y, epochs=500)

print(f"Learned: m={expr.params['m']:.3f} (true: 2.5)")
print(f"Learned: b={expr.params['b']:.3f} (true: 3.0)")
```

## Training a Quadratic

```python
expr = Expression(
    "quad",
    "a*x**2 + b*x + c",
    params={"a": 0.0, "b": 0.0, "c": 0.0},
    trainable_params=["a", "b", "c"]
)

X = np.linspace(-5, 5, 200)
y = 0.5 * X**2 - 2 * X + 1

trainer = Trainer(expr, learning_rate=0.0001)
history = trainer.fit(X, y, epochs=1000)
```

## Using Adam Optimizer

```python
trainer = Trainer(expr, learning_rate=0.01, optimizer="adam")
history = trainer.fit(X, y, epochs=200)
```

## Visualizing Training

```python
from neurogebra.viz.plotting import plot_training_history, plot_expression

# Plot loss curve
fig = plot_training_history(history)

# Plot fitted expression
fig = plot_expression(expr, x_range=(-5, 5))
```

## Using Callbacks

```python
def my_callback(epoch, loss, params):
    if epoch % 50 == 0:
        print(f"Epoch {epoch}: loss={loss:.4f}, params={params}")

trainer = Trainer(expr, learning_rate=0.01)
history = trainer.fit(X, y, epochs=300, callback=my_callback)
```

## Tips for Training

1. **Start with small learning rates** (0.001 or less)
2. **Scale your data** - normalize X and y for better convergence
3. **Use Adam** for faster convergence on complex expressions
4. **Monitor the loss** - if it oscillates, reduce learning rate
5. **Try different initializations** - parameters start at their initial values
