# Custom Loss Functions

Learn how to create and use custom loss functions with Neurogebra.

## Creating a Custom Loss

```python
from neurogebra import Expression, MathForge

# Focal Loss for class imbalance
focal_loss = Expression(
    name="focal_loss",
    symbolic_expr="-alpha * (1 - y_pred)**gamma * y_true * log(y_pred)",
    params={"alpha": 0.25, "gamma": 2.0},
    metadata={
        "category": "loss",
        "description": "Focal Loss for addressing class imbalance",
        "usage": "Object detection, imbalanced classification",
    }
)

# Evaluate
result = focal_loss.eval(y_pred=0.9, y_true=1.0)
print(f"Focal loss: {result}")
```

## Composing Loss Functions

```python
forge = MathForge()

# Weighted combination
mse = forge.get("mse")
mae = forge.get("mae")

# Hybrid loss
hybrid = 0.8 * mse + 0.2 * mae

# String-based composition
composed = forge.compose("mse + 0.1*mae")
```

## Loss with Regularization

```python
from neurogebra.repository.regularizers import get_regularizers

regs = get_regularizers()
l2 = regs["l2"]

# Regularized loss
mse = forge.get("mse")
regularized_loss = mse + 0.01 * l2
```

## Understanding Losses

```python
forge = MathForge()

# Compare MSE vs MAE
print(forge.explain("mse", level="advanced"))
print(forge.explain("mae", level="advanced"))
print(forge.explain("huber", level="advanced"))

# Visual comparison
from neurogebra.viz.plotting import plot_comparison

losses = [forge.get("mse"), forge.get("mae")]
fig = plot_comparison(losses, x_range=(-3, 3))
```
