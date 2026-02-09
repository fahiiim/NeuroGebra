# What is Machine Learning?

This page explains Machine Learning in simple terms. No jargon, no PhD required.

---

## The Simple Explanation

**Machine Learning** is teaching computers to learn from data instead of being explicitly programmed.

### Traditional Programming vs ML

```
Traditional Programming:
    INPUT (data) + RULES (code) → OUTPUT (answers)
    
    Example: if temperature > 30: print("Hot")

Machine Learning:
    INPUT (data) + OUTPUT (answers) → RULES (learned model)
    
    Example: Give the model 1000 examples of temperatures 
             labeled "Hot" or "Cold", and it LEARNS the rule.
```

---

## A Concrete Example

### Without ML: You Write the Rules

```python
def predict_house_price(size, bedrooms, location):
    """Hand-coded rules — fragile and incomplete."""
    base = size * 100
    base += bedrooms * 5000
    if location == "city":
        base *= 1.5
    return base
```

Problems: How did you choose `100`? What about pools? Garages? This doesn't scale.

### With ML: The Computer Learns the Rules

```python
from neurogebra import Expression
from neurogebra.core.trainer import Trainer
import numpy as np

# Define: price = w1*size + w2*bedrooms + bias
# The model will LEARN w1, w2, and bias from data

expr = Expression(
    "house_price",
    "w1*x + w2*x2 + b",
    params={"w1": 0.0, "w2": 0.0, "b": 0.0},
    trainable_params=["w1", "w2", "b"]
)

# Feed it data → it learns the rules automatically
```

---

## The Three Ingredients

Every ML system needs:

### 1. Data

The examples the model learns from.

```python
# Example: House data
sizes =    [800,  1200, 1500, 2000, 2500]
prices =   [150k, 220k, 280k, 350k, 430k]
```

### 2. Model

The mathematical formula with **learnable parameters**.

```python
# price = weight × size + bias
# The model: f(x) = w*x + b
# w and b are unknown → the model will learn them
```

### 3. Training

The process of finding the best parameters by minimizing **error**.

```python
# Step 1: Start with random weights
# Step 2: Make predictions
# Step 3: Measure error (loss)
# Step 4: Adjust weights to reduce error
# Step 5: Repeat steps 2-4 many times (epochs)
```

---

## The Training Loop — The Heart of ML

```
┌──────────────────────────────────────────┐
│          THE TRAINING LOOP               │
│                                          │
│   ┌─────────┐                            │
│   │  START   │ (random weights)          │
│   └────┬────┘                            │
│        ▼                                 │
│   ┌─────────────────┐                    │
│   │ Forward Pass     │ (make prediction) │
│   │ ŷ = w*x + b     │                    │
│   └────────┬────────┘                    │
│            ▼                             │
│   ┌─────────────────┐                    │
│   │ Compute Loss     │ (measure error)   │
│   │ L = (ŷ - y)²    │                    │
│   └────────┬────────┘                    │
│            ▼                             │
│   ┌─────────────────┐                    │
│   │ Backward Pass    │ (compute grads)   │
│   │ dL/dw, dL/db    │                    │
│   └────────┬────────┘                    │
│            ▼                             │
│   ┌─────────────────┐                    │
│   │ Update Weights   │ (learn!)          │
│   │ w = w - lr*dL/dw │                    │
│   └────────┬────────┘                    │
│            │                             │
│            └──── Repeat N epochs ────┘   │
│                                          │
│   Loss gets smaller each time = LEARNING │
└──────────────────────────────────────────┘
```

### In Code (Using Neurogebra):

```python
from neurogebra import Expression
from neurogebra.core.trainer import Trainer
import numpy as np

# The model: y = m*x + b
model = Expression(
    "linear_model",
    "m*x + b",
    params={"m": 0.0, "b": 0.0},
    trainable_params=["m", "b"]
)

# The data (y = 2x + 1)
X = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([3, 5, 7, 9, 11], dtype=float)

# The training
trainer = Trainer(model, learning_rate=0.01)
history = trainer.fit(X, y, epochs=200)

# After training: m ≈ 2.0, b ≈ 1.0
print(f"Learned: y = {model.params['m']:.1f}x + {model.params['b']:.1f}")
```

---

## Key Terms — Your ML Vocabulary

| Term | Simple Meaning | Example |
|------|---------------|---------|
| **Model** | A math formula with adjustable knobs | `y = w*x + b` |
| **Parameters** | The knobs (weights and biases) | `w` and `b` |
| **Training** | Turning the knobs to minimize error | Running 200 epochs |
| **Loss** | How wrong the model is | MSE: `(prediction - actual)²` |
| **Gradient** | Which direction to turn the knobs | `dLoss/dw` (derivative) |
| **Learning Rate** | How much to turn each time | `0.01` — small steps |
| **Epoch** | One complete pass through all data | Epoch 1, 2, 3... 200 |
| **Prediction** | The model's answer | `ŷ = 2.1*3 + 0.9 = 7.2` |
| **Ground Truth** | The correct answer | `y = 7` |

---

## Why It Works

Machine Learning works because of **calculus** — specifically, **gradients (derivatives)**.

The gradient tells the model: "If you increase this weight slightly, the loss will go up/down by this much."

- If the gradient is **positive** → decreasing the weight reduces loss
- If the gradient is **negative** → increasing the weight reduces loss
- If the gradient is **zero** → you're at a minimum (optimal)

Neurogebra makes this visible:

```python
from neurogebra import MathForge

forge = MathForge()
mse = forge.get("mse")

# See the loss function
print(mse.symbolic_expr)  # (y_pred - y_true)**2

# See its gradient
grad = mse.gradient("y_pred")
print(grad.symbolic_expr)  # 2*(y_pred - y_true)

# The gradient tells us: if prediction > true, gradient is positive
# → we should decrease the prediction
# That's exactly what training does!
```

---

**Next:** [Types of Machine Learning →](types-of-ml.md)
