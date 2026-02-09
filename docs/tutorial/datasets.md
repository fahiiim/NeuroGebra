# Datasets

Neurogebra includes built-in **dataset loaders** for common ML tasks, so you can start training immediately without hunting for data.

---

## Loading Datasets

```python
from neurogebra.datasets import loaders

# See what's available
print(dir(loaders))
```

---

## Built-in Datasets

### Synthetic Datasets

These are generated on-the-fly — perfect for learning and testing:

```python
import numpy as np

# === LINEAR DATA ===
# For regression practice
np.random.seed(42)
X = np.linspace(0, 10, 200)
y = 2.5 * X + 3.0 + np.random.normal(0, 1, 200)

# === CLASSIFICATION DATA ===
# Two spirals (classic ML challenge)
def make_spirals(n_points=200, noise=0.5):
    n = n_points // 2
    theta = np.linspace(0, 4*np.pi, n)
    
    r = theta / (4*np.pi)
    x1 = r * np.cos(theta) + np.random.randn(n) * noise * 0.1
    y1 = r * np.sin(theta) + np.random.randn(n) * noise * 0.1
    
    x2 = -r * np.cos(theta) + np.random.randn(n) * noise * 0.1
    y2 = -r * np.sin(theta) + np.random.randn(n) * noise * 0.1
    
    X = np.vstack([np.column_stack([x1, y1]),
                   np.column_stack([x2, y2])])
    y = np.array([0]*n + [1]*n)
    return X, y

X_spiral, y_spiral = make_spirals()
print(f"Spirals: X={X_spiral.shape}, y={y_spiral.shape}")
```

---

## Creating Your Own Datasets

### Regression Data

```python
import numpy as np

def make_regression_data(n=500, noise=0.5, seed=42):
    """Create synthetic regression data."""
    np.random.seed(seed)
    X = np.random.uniform(-5, 5, (n, 1))
    y = 3 * X[:, 0] ** 2 - 2 * X[:, 0] + 1 + np.random.normal(0, noise, n)
    return X, y

X, y = make_regression_data()
print(f"X shape: {X.shape}")  # (500, 1)
print(f"y shape: {y.shape}")  # (500,)
```

### Classification Data

```python
def make_classification_data(n=400, seed=42):
    """Create synthetic binary classification data."""
    np.random.seed(seed)
    
    # Class 0: centered at (-1, -1)
    X0 = np.random.randn(n//2, 2) * 0.8 + [-1, -1]
    
    # Class 1: centered at (1, 1)
    X1 = np.random.randn(n//2, 2) * 0.8 + [1, 1]
    
    X = np.vstack([X0, X1])
    y = np.array([0]*(n//2) + [1]*(n//2))
    
    # Shuffle
    idx = np.random.permutation(n)
    return X[idx], y[idx]

X, y = make_classification_data()
print(f"X shape: {X.shape}")  # (400, 2)
print(f"y shape: {y.shape}")  # (400,)
print(f"Classes: {np.unique(y)}")  # [0 1]
```

---

## Data Splitting

```python
def train_test_split(X, y, test_ratio=0.2, seed=42):
    np.random.seed(seed)
    n = len(X)
    idx = np.random.permutation(n)
    split = int(n * (1 - test_ratio))
    
    return (X[idx[:split]], X[idx[split:]],
            y[idx[:split]], y[idx[split:]])

X_train, X_test, y_train, y_test = train_test_split(X, y)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")
```

---

## Using Datasets with Neurogebra

### Training on Synthetic Data

```python
import numpy as np
from neurogebra import Expression
from neurogebra.core.trainer import Trainer

# Generate data: y = 2x + 1
X = np.linspace(0, 10, 200)
y = 2 * X + 1 + np.random.normal(0, 0.5, 200)

# Create model
model = Expression("linear", "m*x + b",
                   params={"m": 0.0, "b": 0.0},
                   trainable_params=["m", "b"])

# Train
trainer = Trainer(model, learning_rate=0.01, optimizer="adam")
history = trainer.fit(X, y, epochs=200)

print(f"Learned: y = {model.params['m']:.2f}x + {model.params['b']:.2f}")
```

---

## Dataset Best Practices

!!! tip "Tips for Working with Data"

    1. **Always set a random seed** for reproducibility: `np.random.seed(42)`
    2. **Split before processing** — normalize using training data statistics only
    3. **Visualize your data** before training to understand its structure
    4. **Start small** — use small datasets while debugging, scale up later
    5. **Shuffle your data** — ensures model doesn't learn order-dependent patterns

---

**Next:** [Custom Expressions →](../advanced/custom-expressions.md)
