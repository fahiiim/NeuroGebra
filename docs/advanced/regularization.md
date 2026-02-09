# Regularization

Regularization prevents **overfitting** — when a model memorizes training data instead of learning general patterns.

---

## What is Overfitting?

```
Training accuracy: 99%    ← model memorized training data
Test accuracy:     60%    ← fails on new data
                          = OVERFITTING

Training accuracy: 85%
Test accuracy:     83%    ← similar performance
                          = GOOD GENERALIZATION
```

Regularization adds a **penalty** to the loss function that discourages complex models.

---

## Types of Regularization

### L1 Regularization (Lasso)

Pushes some weights to exactly **zero** — acts as feature selection.

$$L_{total} = L_{data} + \lambda \sum |w_i|$$

```python
from neurogebra import MathForge

forge = MathForge()
l1 = forge.get("l1_regularizer")

print(l1.explain())
print(l1.eval(w=0.5, lambda_=0.01))
```

**When to use:** You suspect many features are irrelevant and want automatic feature selection.

---

### L2 Regularization (Ridge)

Pushes all weights toward **small values** — prevents any one weight from dominating.

$$L_{total} = L_{data} + \lambda \sum w_i^2$$

```python
l2 = forge.get("l2_regularizer")

print(l2.explain())
print(l2.eval(w=0.5, lambda_=0.01))
```

**When to use:** All features might be relevant but you want to prevent large weights.

---

### Elastic Net

Combines L1 and L2 — best of both worlds:

$$L_{total} = L_{data} + \lambda_1 \sum |w_i| + \lambda_2 \sum w_i^2$$

```python
elastic = forge.get("elastic_net")

print(elastic.explain())
print(elastic.eval(w=0.5, lambda1=0.01, lambda2=0.01))
```

**When to use:** When you want both feature selection (L1) and small weights (L2).

---

## Comparison

| Type | Effect on Weights | Feature Selection | Best For |
|------|------------------|-------------------|----------|
| **L1 (Lasso)** | Some become 0 | ✅ Yes | Sparse models |
| **L2 (Ridge)** | All become small | ❌ No | Preventing large weights |
| **Elastic Net** | Mix of both | ✅ Partial | General use |

---

## Adding Regularization to Training

### Step-by-Step

```python
import numpy as np
from neurogebra import MathForge, Expression
from neurogebra.core.trainer import Trainer

forge = MathForge()

# 1. Create model
model = Expression(
    "linear_model",
    "w1*x1 + w2*x2 + w3*x3 + b",
    params={"w1": 0.5, "w2": 0.5, "w3": 0.5, "b": 0.0},
    trainable_params=["w1", "w2", "w3", "b"]
)

# 2. Get loss and regularizer
mse = forge.get("mse")
l2 = forge.get("l2_regularizer")

# 3. Create regularized loss (manually)
# total_loss = mse_loss + lambda * sum(w^2)
lambda_reg = 0.01

# 4. Train with the combined loss
trainer = Trainer(model, mse, optimizer="adam", lr=0.01)
```

---

## Regularization Strength ($\lambda$)

The $\lambda$ parameter controls how much regularization to apply:

| $\lambda$ Value | Effect |
|------|--------|
| **0.0** | No regularization (may overfit) |
| **0.001** | Light regularization (usually good start) |
| **0.01** | Moderate regularization |
| **0.1** | Strong regularization (may underfit) |
| **1.0** | Very strong (likely underfitting) |

```python
# Experiment with different lambda values
for lam in [0.0, 0.001, 0.01, 0.1]:
    penalty = l2.eval(w=0.5, lambda_=lam)
    print(f"λ={lam:.3f} → L2 penalty = {penalty:.6f}")
```

**Rule of thumb:** Start with $\lambda = 0.001$ and adjust based on validation performance.

---

## Dropout (Concept)

Another form of regularization — randomly "turns off" neurons during training:

```python
from neurogebra.builders.model_builder import ModelBuilder

builder = ModelBuilder()
model = builder.sequential([
    {"type": "dense", "units": 128, "activation": "relu"},
    {"type": "dropout", "rate": 0.3},       # 30% of neurons turned off randomly
    {"type": "dense", "units": 64, "activation": "relu"},
    {"type": "dropout", "rate": 0.2},       # 20% of neurons turned off
    {"type": "dense", "units": 10, "activation": "softmax"}
])
```

---

## Quick Decision Guide

```
Is your model overfitting?
├── YES → Add regularization
│   ├── Too many features? → Use L1 (Lasso)
│   ├── Weights too large? → Use L2 (Ridge)
│   ├── Not sure? → Use Elastic Net or L2
│   └── Neural network? → Use Dropout + L2
└── NO → You might not need regularization
```

---

**Next:** [Optimization →](optimization.md)
