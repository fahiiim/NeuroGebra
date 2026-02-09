# Loss Functions

A **loss function** measures how wrong your model's predictions are. The goal of training is to **minimize the loss**.

---

## The Concept

```
Model Prediction: ŷ = 4.5
Actual Value:     y  = 5.0

Loss = some_function(ŷ, y)
      = how_wrong_is_the_prediction

If loss is high → model is bad → adjust weights
If loss is low  → model is good → we're done (or close)
```

---

## Exploring Loss Functions

```python
from neurogebra import MathForge

forge = MathForge()
losses = forge.list_all(category="loss")
print(losses)
# ['mse', 'mae', 'binary_crossentropy', 'huber', 'log_cosh', 'hinge', ...]
```

---

## MSE — Mean Squared Error

**The most common loss for regression tasks.**

$$\text{MSE} = (y_{pred} - y_{true})^2$$

```python
mse = forge.get("mse")

# Close prediction → small loss
print(mse.eval(y_pred=4.8, y_true=5.0))  # 0.04

# Bad prediction → large loss
print(mse.eval(y_pred=2.0, y_true=5.0))  # 9.0

# Perfect prediction → zero loss
print(mse.eval(y_pred=5.0, y_true=5.0))  # 0.0
```

**Properties:**

- Squares the error → penalizes large errors heavily
- Always non-negative
- Smooth and differentiable → nice gradients

```python
# The gradient tells the model HOW to adjust
grad = mse.gradient("y_pred")
print(grad.symbolic_expr)  # 2*(y_pred - y_true)
# If pred > true: gradient is positive → decrease prediction
# If pred < true: gradient is negative → increase prediction
```

!!! tip "When to use MSE"
    **Regression tasks.** Best when outliers are rare and you want to penalize large errors.

---

## MAE — Mean Absolute Error

**More robust to outliers than MSE.**

$$\text{MAE} = |y_{pred} - y_{true}|$$

```python
mae = forge.get("mae")

print(mae.eval(y_pred=4.8, y_true=5.0))  # 0.2
print(mae.eval(y_pred=2.0, y_true=5.0))  # 3.0
print(mae.eval(y_pred=5.0, y_true=5.0))  # 0.0
```

**MSE vs MAE:**

| Error | MSE | MAE |
|-------|-----|-----|
| Small (0.2) | 0.04 | 0.2 |
| Medium (3.0) | 9.0 | 3.0 |
| Large (10.0) | 100.0 | 10.0 |

MSE **squares** errors, so it punishes outliers much more. MAE treats all errors linearly.

!!! tip "When to use MAE"
    **Regression with outliers.** More robust than MSE but converges slower.

---

## Binary Cross-Entropy

**The standard loss for binary classification (yes/no problems).**

$$\text{BCE} = -[y_{true} \cdot \log(y_{pred}) + (1-y_{true}) \cdot \log(1-y_{pred})]$$

```python
bce = forge.get("binary_crossentropy")

# Model says 0.9 probability, actual is 1 (correct!) → low loss
print(bce.eval(y_pred=0.9, y_true=1.0))  # ≈ 0.105

# Model says 0.1 probability, actual is 1 (wrong!) → high loss
print(bce.eval(y_pred=0.1, y_true=1.0))  # ≈ 2.303

# Model says 0.9, actual is 0 (wrong!) → high loss
print(bce.eval(y_pred=0.9, y_true=0.0))  # ≈ 2.303
```

!!! tip "When to use Binary Cross-Entropy"
    **Binary classification** (spam/not-spam, disease/healthy). Always pair with sigmoid output.

---

## Huber Loss

**Best of both worlds — MSE for small errors, MAE for large errors.**

$$L_\delta = \begin{cases} \frac{1}{2}(y_{pred} - y_{true})^2 & \text{if } |y_{pred} - y_{true}| \leq \delta \\ \delta|y_{pred} - y_{true}| - \frac{1}{2}\delta^2 & \text{otherwise} \end{cases}$$

```python
huber = forge.get("huber")

# Small error → behaves like MSE (smooth)
# Large error → behaves like MAE (robust)
print(huber.eval(y_pred=4.8, y_true=5.0))  # ≈ 0.02 (MSE-like)
print(huber.eval(y_pred=0.0, y_true=5.0))  # ≈ 4.5  (MAE-like, not 25!)
```

!!! tip "When to use Huber"
    **Regression with occasional outliers.** Great balance between MSE and MAE. Popular in reinforcement learning.

---

## Hinge Loss

**For SVM-style classification with margin.**

$$L = \max(0, 1 - y_{true} \cdot y_{pred})$$

```python
hinge = forge.get("hinge")

# Correct with margin → zero loss
print(hinge.eval(y_pred=2.0, y_true=1.0))   # 0 (correct & confident)

# Correct but no margin → some loss  
print(hinge.eval(y_pred=0.5, y_true=1.0))  # 0.5

# Wrong → high loss
print(hinge.eval(y_pred=-1.0, y_true=1.0))  # 2.0
```

---

## Log-Cosh Loss

**Smooth alternative to Huber Loss.**

$$L = \log(\cosh(y_{pred} - y_{true}))$$

```python
log_cosh = forge.get("log_cosh")

print(log_cosh.eval(y_pred=4.8, y_true=5.0))  # ≈ 0.020
print(log_cosh.eval(y_pred=0.0, y_true=5.0))  # ≈ 4.307
```

---

## Choosing the Right Loss Function

```
What's your task?
│
├── REGRESSION (predicting a number)
│   ├── No outliers       → MSE
│   ├── Has outliers      → MAE or Huber
│   └── Smooth & robust   → Log-Cosh
│
├── BINARY CLASSIFICATION (yes/no)
│   └── → Binary Cross-Entropy + Sigmoid
│
├── MULTI-CLASS CLASSIFICATION (cat/dog/bird)
│   └── → Cross-Entropy + Softmax
│
└── MARGIN-BASED (SVM-style)
    └── → Hinge Loss
```

---

## Comparing Loss Functions

```python
from neurogebra import MathForge
import numpy as np

forge = MathForge()

# Different error levels
errors = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

print(f"{'Error':>6} | {'MSE':>8} | {'MAE':>8} | {'Huber':>8}")
print("-" * 40)

for err in errors:
    mse_val = forge.get("mse").eval(y_pred=err, y_true=0)
    mae_val = forge.get("mae").eval(y_pred=err, y_true=0)
    # Huber needs delta param
    huber_val = min(0.5 * err**2, abs(err) - 0.5)  # delta=1
    
    print(f"{err:>6.1f} | {mse_val:>8.3f} | {mae_val:>8.3f} | {huber_val:>8.3f}")
```

---

## Composing Custom Losses

```python
# Weighted combination
hybrid = forge.compose("mse + 0.1*mae")

# Evaluate
result = hybrid.eval(y_pred=3.0, y_true=5.0)
print(f"Hybrid loss: {result}")
```

---

## Try It Yourself!

!!! example "Exercise"
    1. Compute MSE, MAE, and Huber loss for predictions `[1, 2, 3]` vs targets `[1.5, 2.5, 3.5]`
    2. Which loss function is most sensitive to the error `(pred=0, true=10)`?
    3. Create a custom weighted loss: `0.8*mse + 0.2*mae`
    4. Compute the gradient of MSE with respect to `y_pred`

---

**Next:** [Gradients & Differentiation →](gradients.md)
