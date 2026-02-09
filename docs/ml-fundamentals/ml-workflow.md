# The ML Workflow

Every ML project follows the same steps. This page teaches you the standard workflow that professionals use.

---

## The 7 Steps

```
Step 1: Define the Problem
    ↓
Step 2: Collect & Prepare Data
    ↓
Step 3: Choose a Model
    ↓
Step 4: Train the Model
    ↓
Step 5: Evaluate the Model
    ↓
Step 6: Tune & Improve
    ↓
Step 7: Deploy (Use It)
```

---

## Step 1: Define the Problem

Before writing any code, ask:

- **What am I predicting?** (a number? a category?)
- **What data do I have?** (features and labels?)
- **How will I measure success?** (accuracy? error?)

!!! example "Example"
    **Problem:** Predict house prices from size.
    
    - **Type:** Regression (predicting a number)
    - **Input:** House size in sqft
    - **Output:** Price in dollars
    - **Success metric:** Mean Squared Error (lower = better)

---

## Step 2: Collect & Prepare Data

```python
import numpy as np

# Simulate a dataset
np.random.seed(42)
n = 200

# Features (inputs)
house_size = np.random.uniform(500, 3000, n)

# Labels (outputs) — the "ground truth"
price = 150 * house_size + 50000 + np.random.normal(0, 20000, n)

# Split into train/test
split = int(0.8 * n)
X_train, X_test = house_size[:split], house_size[split:]
y_train, y_test = price[:split], price[split:]

# Normalize features
X_mean, X_std = X_train.mean(), X_train.std()
X_train_norm = (X_train - X_mean) / X_std
X_test_norm = (X_test - X_mean) / X_std

print(f"Training samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")
```

---

## Step 3: Choose a Model

Start simple. You can always make it more complex later.

```python
from neurogebra import Expression

# Simple linear model: price = w * size + b
model = Expression(
    "house_price_model",
    "w * x + b",
    params={"w": 0.0, "b": 0.0},
    trainable_params=["w", "b"]
)
```

!!! tip "Model Selection Rule"
    - **Linear data?** → Start with linear model
    - **Non-linear data?** → Use polynomial or neural network
    - **Image data?** → Use CNN (convolutional neural network)
    - **Sequence data?** → Use RNN or Transformer

---

## Step 4: Train the Model

```python
from neurogebra.core.trainer import Trainer

trainer = Trainer(model, learning_rate=0.01, optimizer="adam")

history = trainer.fit(
    X_train_norm, y_train,
    epochs=300,
    verbose=True
)

print(f"\nLearned parameters:")
print(f"  w = {model.params['w']:.2f}")
print(f"  b = {model.params['b']:.2f}")
```

### What Happens During Training?

| Epoch | Loss | What's Happening |
|-------|------|-----------------|
| 0 | 1,000,000 | Random weights — terrible predictions |
| 50 | 500,000 | Getting better — weights adjusting |
| 100 | 200,000 | Much better — model is learning the pattern |
| 200 | 100,000 | Good — close to optimal |
| 300 | 95,000 | Converged — further training won't help much |

---

## Step 5: Evaluate the Model

**Never evaluate on training data.** Use the test set:

```python
# Predict on test data
predictions = np.array([model.eval(x=xi) for xi in X_test_norm])

# Calculate metrics
mse = np.mean((predictions - y_test) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(predictions - y_test))

print(f"Test MSE:  {mse:.2f}")
print(f"Test RMSE: {rmse:.2f}")
print(f"Test MAE:  {mae:.2f}")

# R² score (how much variance is explained)
ss_res = np.sum((y_test - predictions) ** 2)
ss_tot = np.sum((y_test - y_test.mean()) ** 2)
r2 = 1 - ss_res / ss_tot
print(f"R² Score: {r2:.4f}")  # 1.0 = perfect, 0.0 = terrible
```

### Understanding the Metrics

| Metric | What It Measures | Good Value |
|--------|-----------------|------------|
| MSE | Average squared error | Lower = better |
| RMSE | Average error (same units as data) | Lower = better |
| MAE | Average absolute error | Lower = better |
| R² | How much variance is explained | 1.0 = perfect |

---

## Step 6: Tune & Improve

If the model isn't good enough, try:

### Change Learning Rate

```python
# Too high → oscillates, might diverge
trainer = Trainer(model, learning_rate=1.0)  # Bad!

# Too low → takes forever to converge
trainer = Trainer(model, learning_rate=0.00001)  # Slow!

# Just right → smooth convergence
trainer = Trainer(model, learning_rate=0.01)  # Good!
```

### Use a Better Optimizer

```python
# Adam usually works better than SGD
trainer = Trainer(model, learning_rate=0.01, optimizer="adam")
```

### Use a More Complex Model

```python
# Polynomial model for non-linear data
model = Expression(
    "polynomial",
    "a*x**2 + b*x + c",
    params={"a": 0.0, "b": 0.0, "c": 0.0},
    trainable_params=["a", "b", "c"]
)
```

### Train Longer

```python
trainer.fit(X_train_norm, y_train, epochs=1000)
```

---

## Step 7: Deploy (Use It)

Once your model is good, use it to make predictions on new data:

```python
# New house: 1800 sqft
new_size = 1800
new_size_norm = (new_size - X_mean) / X_std

predicted_price = model.eval(x=new_size_norm)
print(f"Predicted price for {new_size} sqft: ${predicted_price:,.2f}")
```

---

## Common Pitfalls

!!! warning "Overfitting"
    **Problem:** Model memorizes training data but fails on new data.
    
    **Signs:** Training loss is very low, test loss is high.
    
    **Solutions:** More data, regularization, simpler model, dropout.

!!! warning "Underfitting"
    **Problem:** Model is too simple to capture the pattern.
    
    **Signs:** Both training and test loss are high.
    
    **Solutions:** More complex model, more features, train longer.

!!! warning "Data Leakage"
    **Problem:** Test data accidentally influences training.
    
    **Signs:** Suspiciously good test results.
    
    **Solutions:** Always split data BEFORE any processing.

---

## Complete Workflow Example

```python
import numpy as np
from neurogebra import Expression
from neurogebra.core.trainer import Trainer

# === STEP 1: Define ===
# Predicting y = 3x + 2 from noisy data

# === STEP 2: Data ===
np.random.seed(42)
X = np.linspace(0, 10, 200)
y = 3 * X + 2 + np.random.normal(0, 1, 200)

X_train, X_test = X[:160], X[160:]
y_train, y_test = y[:160], y[160:]

# === STEP 3: Model ===
model = Expression("linear", "m*x + b",
                   params={"m": 0.0, "b": 0.0},
                   trainable_params=["m", "b"])

# === STEP 4: Train ===
trainer = Trainer(model, learning_rate=0.001, optimizer="adam")
history = trainer.fit(X_train, y_train, epochs=500, verbose=True)

# === STEP 5: Evaluate ===
preds = np.array([model.eval(x=xi) for xi in X_test])
mse = np.mean((preds - y_test) ** 2)
print(f"\nTest MSE: {mse:.4f}")
print(f"Learned: y = {model.params['m']:.2f}x + {model.params['b']:.2f}")
# Expected: y ≈ 3.00x + 2.00
```

---

**Next:** [Math Behind ML →](math-behind-ml.md)
