# Optimization

Optimizers control **how** your model updates its weights during training. Choosing the right optimizer can be the difference between a model that converges in 10 epochs vs. one that never learns.

---

## What is Optimization?

Training = finding the weights that minimize the loss function.

```
Current weights →  Compute loss  →  Compute gradients  →  Update weights
      ↑                                                          |
      └──────────────────────────────────────────────────────────┘
      
The OPTIMIZER decides how to use the gradients to update weights.
```

---

## SGD (Stochastic Gradient Descent)

The simplest optimizer: move in the direction opposite to the gradient.

$$w_{new} = w_{old} - \eta \cdot \nabla L$$

Where $\eta$ is the **learning rate**.

```python
from neurogebra import MathForge, Expression
from neurogebra.core.trainer import Trainer
import numpy as np

forge = MathForge()

model = Expression("linear", "w*x + b",
    params={"w": 0.0, "b": 0.0}, trainable_params=["w", "b"])
loss = forge.get("mse")

# SGD optimizer
trainer = Trainer(model, loss, optimizer="sgd", lr=0.01)

X = np.linspace(0, 10, 50)
y = 3 * X + 2 + np.random.normal(0, 0.5, 50)
history = trainer.fit(X, y, epochs=100)
```

### Pros and Cons

| Pros | Cons |
|---------|---------|
| Simple to understand | Can be slow to converge |
| Low memory usage | Sensitive to learning rate |
| Good for convex problems | Gets stuck in local minima |

---

## Adam (Adaptive Moment Estimation)

The most popular optimizer. It adapts the learning rate for each parameter individually.

Adam combines:

- **Momentum**: Remembers past gradients (like a ball rolling downhill)
- **Adaptive learning rate**: Different learning rates for different parameters

```python
# Adam optimizer (recommended for most cases)
trainer = Trainer(model, loss, optimizer="adam", lr=0.001)
history = trainer.fit(X, y, epochs=100)
```

### Pros and Cons

| Pros | Cons |
|---------|---------|
| Fast convergence | Slightly more memory |
| Works well out-of-the-box | May not generalize as well as SGD |
| Handles sparse gradients | Can overshoot minimum |
| Good default for most problems | Sensitive to epsilon on noisy inputs |

---

## SGD vs Adam: When to Use What

| Scenario | Recommended | Why |
|----------|-------------|-----|
| First attempt | **Adam** | Works well with defaults |
| Simple regression | **SGD** | Sufficient for convex problems |
| Deep neural networks | **Adam** | Handles complex loss landscapes |
| Need best generalization | **SGD** (with tuning) | Often finds flatter minima |
| Quick prototyping | **Adam** | Less hyperparameter tuning |

---

## Learning Rate

The most important hyperparameter. Controls step size:

$$w_{new} = w_{old} - \underbrace{\eta}_{\text{learning rate}} \cdot \nabla L$$

```python
import numpy as np
from neurogebra import Expression
from neurogebra.core.trainer import Trainer

# Too small: slow convergence
trainer_slow = Trainer(model, loss, optimizer="adam", lr=0.0001)

# Just right: converges nicely
trainer_good = Trainer(model, loss, optimizer="adam", lr=0.001)

# Too large: overshoots, loss bounces
trainer_fast = Trainer(model, loss, optimizer="adam", lr=0.1)
```

### Learning Rate Effects

```
Learning Rate Too Small:
Loss: ████████████████████████░░ (barely decreasing after 1000 epochs)

Learning Rate Just Right:
Loss: ████████████░░░░░░░░░░░░░░ (steadily decreasing, converges at ~200 epochs)

Learning Rate Too Large:
Loss: ████████████████████████████ (bouncing around, never converges)
```

### Recommended Starting Points

| Optimizer | Starting LR | Range to Try |
|-----------|-------------|--------------|
| **SGD** | 0.01 | 0.001 - 0.1 |
| **Adam** | 0.001 | 0.0001 - 0.01 |

---

## Manual Gradient Descent

For understanding, you can implement optimization manually:

```python
from neurogebra.core.autograd import Value
import numpy as np

# Simple optimization: find x that minimizes x^2 - 4x + 4  (answer: x=2)
x = Value(0.0)  # Start at x=0
lr = 0.1

for epoch in range(50):
    # Forward pass
    loss = x**2 - 4*x + 4
    
    # Backward pass
    loss.backward()
    
    # Update
    x.data -= lr * x.grad
    
    # Zero gradients
    x.grad = 0.0
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: x = {x.data:.4f}, loss = {loss.data:.4f}")

print(f"\nOptimal x = {x.data:.4f}")  # Should be ≈ 2.0
```

---

## Batch Size and Optimization

How much data you use per update also matters:

```python
# Full batch: use all data per update
trainer = Trainer(model, loss, optimizer="adam", lr=0.001)
history = trainer.fit(X, y, epochs=100, batch_size=len(X))

# Mini-batch: use 32 samples per update (most common)
history = trainer.fit(X, y, epochs=100, batch_size=32)

# Stochastic: use 1 sample per update
history = trainer.fit(X, y, epochs=100, batch_size=1)
```

| Batch Size | Speed | Stability | Memory |
|-----------|-------|-----------|--------|
| **Full batch** | Slow per epoch | Stable | High |
| **Mini-batch (32)** | Good balance | Moderate noise | Moderate |
| **Single (1)** | Fast per epoch | Very noisy | Low |

**Recommendation:** Start with batch_size=32.

---

## Standalone Optimizers (`neurogebra.optim`)

NeuroGebra provides standalone optimizer objects with full momentum, weight decay, and Nesterov acceleration that can be passed directly to `model.compile()` or used in custom training loops.

```python
from neurogebra.optim import SGD, Adam, RMSprop

# SGD with momentum and weight decay
opt_sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True, weight_decay=1e-4)

# Adam with custom beta parameters
opt_adam = Adam(learning_rate=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)

# RMSprop
opt_rmsprop = RMSprop(learning_rate=0.001, alpha=0.99, eps=1e-8)

# Pass directly to ModelBuilder compiled model:
model.compile(loss="categorical_crossentropy", optimizer=opt_adam)
```

---

## Learning Rate Schedulers

Dynamic learning rate adjustment during training prevents premature convergence and helps models settle into sharper minima. NeuroGebra includes four production-ready schedulers in `neurogebra.optim`:

### 1. StepLR
Decays the learning rate by `gamma` every `step_size` epochs:

$$\eta_t = \eta_0 \cdot \gamma^{\lfloor t / \text{step\_size} \rfloor}$$

```python
from neurogebra.optim import StepLR

scheduler = StepLR(model, step_size=10, gamma=0.5)
history = model.fit(X, y, epochs=50, batch_size=32, lr_scheduler=scheduler)
```

### 2. CosineAnnealingLR
Anneals the learning rate following a cosine curve down to `eta_min`:

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_0 - \eta_{\min})\left(1 + \cos\left(\frac{t}{T_{\max}}\pi\right)\right)$$

```python
from neurogebra.optim import CosineAnnealingLR

scheduler = CosineAnnealingLR(model, T_max=50, eta_min=1e-5)
history = model.fit(X, y, epochs=50, batch_size=32, lr_scheduler=scheduler)
```

### 3. ExponentialLR
Decays the learning rate exponentially at every epoch:

$$\eta_t = \eta_0 \cdot \gamma^t$$

```python
from neurogebra.optim import ExponentialLR

scheduler = ExponentialLR(model, gamma=0.95)
history = model.fit(X, y, epochs=50, batch_size=32, lr_scheduler=scheduler)
```

### 4. ReduceLROnPlateau
Monitors validation or training loss and reduces the learning rate when progress stalls:

```python
from neurogebra.optim import ReduceLROnPlateau

scheduler = ReduceLROnPlateau(model, mode="min", factor=0.5, patience=5, min_lr=1e-6)
# Updated automatically by model.fit when val_loss plateaus
history = model.fit(X, y, epochs=50, batch_size=32, lr_scheduler=scheduler)
```

---

## Optimization Tips

1. **Start with Adam, lr=0.001** — works for 90% of cases
2. **Watch the loss curve** — should decrease smoothly
3. **If loss plateaus** — try reducing learning rate by 10x
4. **If loss bounces** — reduce learning rate
5. **If loss NaN** — learning rate is way too high
6. **Train longer if needed** — some problems need more epochs

---

**Next:** [Performance Tips →](performance.md)
