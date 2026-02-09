# Training Expressions

This is where the magic happens — **teaching expressions to learn from data**.

---

## The Concept

"Training" means finding the parameter values that make the expression match the data as closely as possible.

```
Before training: y = 0.0*x + 0.0  (random — useless)
After training:  y = 2.0*x + 1.0  (learned from data!)
```

---

## Step-by-Step: Training a Linear Model

### Step 1: Create a Trainable Expression

```python
from neurogebra import Expression

model = Expression(
    "linear_model",
    "m*x + b",
    params={"m": 0.0, "b": 0.0},        # Start with zeros
    trainable_params=["m", "b"]           # These will be learned
)

print(f"Before training: y = {model.params['m']}*x + {model.params['b']}")
# Before training: y = 0.0*x + 0.0
```

### Step 2: Prepare Data

```python
import numpy as np

# True relationship: y = 2x + 1
np.random.seed(42)
X = np.linspace(0, 10, 100)
y = 2 * X + 1 + np.random.normal(0, 0.5, 100)  # Add some noise
```

### Step 3: Create a Trainer

```python
from neurogebra.core.trainer import Trainer

trainer = Trainer(
    model,
    learning_rate=0.01,   # How big each adjustment step is
    optimizer="sgd"        # Stochastic Gradient Descent
)
```

### Step 4: Train!

```python
history = trainer.fit(
    X, y,
    epochs=200,       # How many times to loop through the data
    verbose=True       # Print progress
)
```

Output:
```
Epoch    0/200: Loss = 25.431200
Epoch   20/200: Loss = 3.214100
Epoch   40/200: Loss = 0.891230
Epoch   60/200: Loss = 0.412340
...
Epoch  200/200: Loss = 0.251230
```

### Step 5: Check Results

```python
print(f"After training: y = {model.params['m']:.2f}*x + {model.params['b']:.2f}")
# After training: y = 2.01*x + 0.98
# Very close to the true y = 2x + 1!
```

---

## Understanding the Trainer

### Optimizers

| Optimizer | Description | When to Use |
|-----------|-------------|-------------|
| `"sgd"` | Stochastic Gradient Descent | Simple, educational, basic tasks |
| `"adam"` | Adaptive Moment Estimation | Default choice, works almost always |

```python
# SGD — simple but sometimes slow
trainer_sgd = Trainer(model, learning_rate=0.01, optimizer="sgd")

# Adam — adaptive learning rate, usually faster
trainer_adam = Trainer(model, learning_rate=0.01, optimizer="adam")
```

### Learning Rate

The learning rate controls how much parameters change each step:

```python
# Too high → overshoots, loss oscillates or explodes
trainer = Trainer(model, learning_rate=1.0)    # Bad!

# Too low → takes forever to converge
trainer = Trainer(model, learning_rate=0.00001)  # Very slow!

# Just right → smooth convergence
trainer = Trainer(model, learning_rate=0.01)    # Good starting point
```

### Loss Functions

```python
# Default: MSE (mean squared error)
history = trainer.fit(X, y, loss_fn="mse")

# Alternative: MAE (mean absolute error)
history = trainer.fit(X, y, loss_fn="mae")

# Alternative: Huber (robust to outliers)
history = trainer.fit(X, y, loss_fn="huber")
```

### Mini-Batch Training

```python
# Full batch (default) — uses all data each step
history = trainer.fit(X, y, batch_size=None)

# Mini-batch — uses small chunks (faster for large datasets)
history = trainer.fit(X, y, batch_size=32)
```

---

## Training History

The `fit()` method returns a history dictionary:

```python
history = trainer.fit(X, y, epochs=100)

# Loss over time
print(history["loss"][:5])   # First 5 losses
print(history["loss"][-5:])  # Last 5 losses

# Parameters over time
print(history["params"][0])    # Parameters at epoch 0
print(history["params"][-1])   # Parameters at last epoch
```

---

## Example: Training a Quadratic Model

```python
import numpy as np
from neurogebra import Expression
from neurogebra.core.trainer import Trainer

# True function: y = x² - 2x + 1
X = np.linspace(-3, 3, 100)
y = X**2 - 2*X + 1 + np.random.normal(0, 0.3, 100)

# Model with unknown coefficients
model = Expression(
    "quadratic",
    "a*x**2 + b*x + c",
    params={"a": 0.0, "b": 0.0, "c": 0.0},
    trainable_params=["a", "b", "c"]
)

# Train
trainer = Trainer(model, learning_rate=0.001, optimizer="adam")
history = trainer.fit(X, y, epochs=500, verbose=True)

print(f"\nLearned: y = {model.params['a']:.2f}x² + ({model.params['b']:.2f})x + {model.params['c']:.2f}")
# Expected: y ≈ 1.00x² + (-2.00)x + 1.00
```

---

## Example: Using a Callback

```python
def my_callback(epoch, loss, params):
    """Called after each epoch."""
    if loss < 0.5:
        print(f"  [Early stop possible] Epoch {epoch}: loss = {loss:.4f}")

trainer = Trainer(model, learning_rate=0.01, optimizer="adam")
history = trainer.fit(X, y, epochs=200, callback=my_callback)
```

---

## Training Tips

!!! tip "Best Practices"

    **Start with Adam optimizer** — it handles most situations well.
    
    **Use learning rate 0.01** as a starting point. If loss oscillates, decrease it. If loss decreases too slowly, increase it.
    
    **Watch the loss curve:**
    
    - Smooth decrease → good
    - Oscillating → learning rate too high
    - Flat (no decrease) → learning rate too low or model too simple
    - Sudden explosion → learning rate WAY too high
    
    **Normalize your data** before training.
    
    **Use enough epochs** — but not too many (overfitting risk).

---

## Complete Training Pipeline

```python
import numpy as np
from neurogebra import Expression
from neurogebra.core.trainer import Trainer

# 1. Generate data
np.random.seed(42)
X = np.linspace(-5, 5, 200)
y = 0.5 * X**2 + 2 * X - 3 + np.random.normal(0, 1, 200)

# 2. Split data (80% train, 20% test)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 3. Define model
model = Expression(
    "polynomial",
    "a*x**2 + b*x + c",
    params={"a": 0.0, "b": 0.0, "c": 0.0},
    trainable_params=["a", "b", "c"]
)

# 4. Train
trainer = Trainer(model, learning_rate=0.001, optimizer="adam")
history = trainer.fit(X_train, y_train, epochs=500, verbose=True)

# 5. Evaluate on test set
predictions = np.array([model.eval(x=xi) for xi in X_test])
test_mse = np.mean((predictions - y_test) ** 2)

print(f"\nLearned: y = {model.params['a']:.2f}x² + {model.params['b']:.2f}x + {model.params['c']:.2f}")
print(f"Test MSE: {test_mse:.4f}")
print(f"True:     y = 0.50x² + 2.00x - 3.00")
```

---

**Next:** [Autograd Engine →](autograd.md)
