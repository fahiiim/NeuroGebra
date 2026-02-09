# Types of Machine Learning

Machine Learning has three main categories. Each solves a different kind of problem.

---

## 1. Supervised Learning

**You have answers (labels) for your training data.**

The model learns the relationship between inputs and outputs.

```
Training Data:
    Input: house_size = 1500 sqft → Output: price = $280,000 ✓
    Input: house_size = 2000 sqft → Output: price = $350,000 ✓
    Input: house_size = 800 sqft  → Output: price = $150,000 ✓

After Training:
    Input: house_size = 1800 sqft → Output: price = ???
    Model predicts: $315,000
```

### Two Sub-Types:

#### Regression — Predicting Numbers

Output is a **continuous value** (price, temperature, score).

```python
from neurogebra import Expression
from neurogebra.core.trainer import Trainer
import numpy as np

# REGRESSION: Predict a number
model = Expression("linear", "m*x + b",
                   params={"m": 0.0, "b": 0.0},
                   trainable_params=["m", "b"])

X = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([2.1, 3.9, 6.1, 7.8, 10.2], dtype=float)

trainer = Trainer(model, learning_rate=0.01)
trainer.fit(X, y, epochs=200)
```

| Use Cases | Loss Functions | Output |
|-----------|---------------|--------|
| House prices | MSE, MAE | Continuous number |
| Stock prediction | Huber | Continuous number |
| Temperature forecast | MSE | Continuous number |

#### Classification — Predicting Categories

Output is a **class/category** (spam/not-spam, cat/dog/bird).

```python
# The model outputs a probability using Sigmoid
from neurogebra import MathForge

forge = MathForge()
sigmoid = forge.get("sigmoid")

# Output > 0.5 → Class 1 (spam)
# Output < 0.5 → Class 0 (not spam)
print(sigmoid.eval(x=2.0))   # 0.88 → "It's spam"
print(sigmoid.eval(x=-1.5))  # 0.18 → "Not spam"
```

| Use Cases | Loss Functions | Output |
|-----------|---------------|--------|
| Spam detection | Binary Cross-Entropy | 0 or 1 |
| Image recognition | Cross-Entropy | Class label |
| Disease diagnosis | Binary Cross-Entropy | Yes or No |

---

## 2. Unsupervised Learning

**You have NO answers (labels). The model finds patterns on its own.**

```
Data (no labels):
    [1.2, 1.3], [1.1, 1.4], [5.5, 5.6], [5.2, 5.8], [9.0, 9.1]

Model finds:
    Cluster A: [1.2, 1.3], [1.1, 1.4]         (small values)
    Cluster B: [5.5, 5.6], [5.2, 5.8]         (medium values)
    Cluster C: [9.0, 9.1]                      (large values)
```

| Use Cases | Algorithms |
|-----------|-----------|
| Customer segmentation | K-Means clustering |
| Anomaly detection | Autoencoders |
| Data compression | PCA |
| Recommendation | Collaborative filtering |

---

## 3. Reinforcement Learning

**The model learns by trial and error, receiving rewards or penalties.**

```
Environment: Video game
Agent: The AI player
Actions: Move left, right, jump
Rewards: +10 for collecting coin, -100 for falling

The agent tries random actions, learns which ones lead to rewards,
and eventually masters the game.
```

| Use Cases | Examples |
|-----------|---------|
| Game playing | AlphaGo, Atari games |
| Robotics | Walking, grasping objects |
| Self-driving | Navigation decisions |

---

## Comparison Table

| Type | Has Labels? | Goal | Neurogebra Focus |
|------|------------|------|-----------------|
| Supervised | Yes | Predict from labeled data | ✅ Full support |
| Unsupervised | No | Find patterns | Partial |
| Reinforcement | Rewards | Maximize reward | Future feature |

!!! info "Neurogebra focuses on Supervised Learning"
    This is where you'll spend most of your ML journey. Neurogebra excels at making supervised learning **understandable** — you can see every loss function, every gradient, every optimization step.

---

## The ML Problem Decision Tree

```
What kind of output do you need?
│
├── A number (price, score, amount) → REGRESSION
│   └── Neurogebra: use MSE/MAE loss, linear output
│
├── A category (yes/no) → BINARY CLASSIFICATION
│   └── Neurogebra: use binary_crossentropy, sigmoid output
│
├── A category (cat/dog/bird) → MULTI-CLASS CLASSIFICATION
│   └── Neurogebra: use cross_entropy, softmax output
│
├── Groups in data (no labels) → CLUSTERING
│   └── Use: K-Means, DBSCAN
│
└── Sequential decisions → REINFORCEMENT LEARNING
    └── Use: Q-Learning, Policy Gradient
```

---

## Neurogebra Makes It Clear

```python
from neurogebra import MathForge

forge = MathForge()

# See all loss functions — pick the right one for your task
losses = forge.list_all(category="loss")
print(losses)

# Each one explains when to use it
for name in ["mse", "mae", "binary_crossentropy", "hinge"]:
    expr = forge.get(name)
    print(f"\n{name}:")
    print(f"  {expr.metadata.get('description', '')}")
    print(f"  Best for: {expr.metadata.get('usage', '')}")
```

---

**Next:** [The ML Workflow →](ml-workflow.md)
