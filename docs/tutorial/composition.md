# Expression Composition

One of Neurogebra's most powerful features is the ability to **combine simple expressions into complex ones**.

---

## Why Compose?

Real ML involves complex math built from simple pieces:

- Custom loss = MSE + regularization
- Complex activation = combination of simple activations
- Neural network layer = linear transformation + activation

---

## Arithmetic Composition

### Addition

```python
from neurogebra import MathForge

forge = MathForge()

mse = forge.get("mse")
mae = forge.get("mae")

# Combine losses
hybrid_loss = mse + mae
result = hybrid_loss.eval(y_pred=3.0, y_true=5.0)
print(f"Hybrid loss: {result}")  # MSE(4) + MAE(2) = 6
```

### Scalar Multiplication

```python
# Weighted loss: 70% MSE + 30% MAE
weighted_loss = 0.7 * mse + 0.3 * mae
```

### String-Based Composition

```python
# Quick composition from string
loss = forge.compose("mse + 0.1*mae")
print(loss.eval(y_pred=3.0, y_true=5.0))
```

---

## Functional Composition

Compose expressions like mathematical function composition: $f(g(x))$

```python
from neurogebra import Expression, MathForge

forge = MathForge()

# Create a linear transformation
linear = Expression("linear", "2*x + 1")

# Compose with sigmoid: sigmoid(2*x + 1)
sigmoid = forge.get("sigmoid")
composed = sigmoid.compose(linear)

print(f"Formula: {composed.symbolic_expr}")

# Evaluate
print(f"f(0) = {composed.eval(x=0):.4f}")  # sigmoid(1) ≈ 0.7311
print(f"f(1) = {composed.eval(x=1):.4f}")  # sigmoid(3) ≈ 0.9526
```

---

## Building Custom Activations

```python
from neurogebra import Expression, MathForge

forge = MathForge()

# Parametric ReLU: max(alpha*x, x)
prelu = Expression(
    "parametric_relu",
    "Max(alpha*x, x)",
    params={"alpha": 0.1},
    trainable_params=["alpha"],  # alpha can be learned!
    metadata={"category": "activation"}
)

print(prelu.eval(x=5))     # 5
print(prelu.eval(x=-5))    # -0.5 (with alpha=0.1)

# Register it for reuse
forge.register("prelu", prelu)
```

---

## Building Custom Loss Functions

```python
# Focal Loss (used in object detection)
# It down-weights easy examples and focuses on hard ones
focal = Expression(
    "focal_loss",
    "-(alpha * y_true * (1-y_pred)**gamma * log(y_pred) + (1-alpha) * (1-y_true) * y_pred**gamma * log(1-y_pred))",
    params={"alpha": 0.25, "gamma": 2.0},
    metadata={
        "category": "loss",
        "description": "Focal loss for imbalanced classification"
    }
)

forge.register("focal_loss", focal)
```

---

## Building Regularized Losses

Regularization prevents overfitting by penalizing large weights:

```python
from neurogebra import MathForge, Expression

forge = MathForge()

# Base loss
mse = forge.get("mse")

# L2 regularizer from repository
l2 = Expression("l2_term", "lambda_reg * w**2",
                params={"lambda_reg": 0.01})

# Regularized loss = MSE + L2
# In practice, you'd sum L2 over all weights
print("Base MSE:", mse.eval(y_pred=3, y_true=5))
print("L2 penalty:", l2.eval(w=2.0))     # 0.01 * 4 = 0.04
```

---

## Multi-Step Composition

Build complex pipelines step by step:

```python
from neurogebra import Expression

# Step 1: Normalize input
normalize = Expression("normalize", "(x - mean) / std",
                       params={"mean": 0.0, "std": 1.0})

# Step 2: Linear transformation
linear = Expression("linear", "w * x + b",
                    params={"w": 1.0, "b": 0.0})

# Step 3: Activation
from neurogebra import MathForge
forge = MathForge()
relu = forge.get("relu")

# Chain them: relu(w * normalize(x) + b)
step1 = linear.compose(normalize)   # w * ((x-mean)/std) + b
final = relu.compose(step1)         # relu(above)

print(f"Pipeline: {final.symbolic_expr}")
```

---

## Practical Example: Custom Model

```python
from neurogebra import Expression, MathForge
import numpy as np

forge = MathForge()

# Build a simple 2-layer network expression
# Layer 1: h = relu(w1*x + b1)
# Layer 2: y = w2*h + b2

# We can compose this step by step
layer1_linear = Expression("l1", "w1*x + b1",
                           params={"w1": 0.5, "b1": 0.0})

relu = forge.get("relu")
layer1 = relu.compose(layer1_linear)  # relu(w1*x + b1)

print(f"Layer 1 output at x=2: {layer1.eval(x=2)}")
# relu(0.5*2 + 0) = relu(1.0) = 1.0
```

---

## Tips for Composition

!!! tip "Best Practices"
    1. **Start simple** — compose from well-tested building blocks
    2. **Test each step** — evaluate intermediate expressions
    3. **Register reusable expressions** — `forge.register("name", expr)`
    4. **Use string composition for quick experiments** — `forge.compose("mse + 0.1*mae")`
    5. **Check gradients** — composed expressions have gradients too!

---

**Next:** [Training Expressions →](training.md)
