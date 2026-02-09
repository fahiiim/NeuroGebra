# Performance Tips

Tips to make your Neurogebra code run faster and use less memory.

---

## 1. Use NumPy Arrays, Not Python Lists

```python
import numpy as np
from neurogebra import MathForge

forge = MathForge()
relu = forge.get("relu")

# ❌ SLOW: Python list
x_list = [1, 2, 3, 4, 5]
result = [relu.eval(x=val) for val in x_list]

# ✅ FAST: NumPy array (vectorized)
x_array = np.array([1, 2, 3, 4, 5])
result = relu.eval(x=x_array)
```

NumPy arrays are **100-1000x faster** for mathematical operations.

---

## 2. Batch Evaluation

Evaluate many points at once instead of one at a time:

```python
from neurogebra import Expression
import numpy as np

model = Expression("quadratic", "a*x**2 + b*x + c",
    params={"a": 1.0, "b": -2.0, "c": 1.0})

# ❌ SLOW: One at a time
results = []
for x_val in range(1000):
    results.append(model.eval(x=x_val))

# ✅ FAST: All at once
x = np.arange(1000)
results = model.eval(x=x)
```

---

## 3. Pre-compute Gradients

If you need the same gradient repeatedly, compute it once:

```python
from neurogebra import MathForge

forge = MathForge()
sigmoid = forge.get("sigmoid")

# ❌ SLOW: Compute gradient each time
for x_val in range(100):
    grad = sigmoid.gradient("x")      # Recalculates symbolic gradient each time
    result = grad.eval(x=float(x_val))

# ✅ FAST: Compute gradient once, evaluate many times
grad = sigmoid.gradient("x")          # Compute once
x = np.arange(100, dtype=float)
results = grad.eval(x=x)              # Evaluate all at once
```

---

## 4. Use Appropriate Batch Sizes for Training

```python
from neurogebra.core.trainer import Trainer

# For small datasets (< 1000 samples)
# Use full batch — no overhead from splitting
trainer.fit(X, y, epochs=100, batch_size=len(X))

# For medium datasets (1000 - 100000 samples)
# Use mini-batches of 32-128
trainer.fit(X, y, epochs=100, batch_size=32)

# For large datasets (> 100000 samples)
# Use mini-batches of 64-256
trainer.fit(X, y, epochs=50, batch_size=128)
```

---

## 5. Use the Right Optimizer

Adam converges faster than SGD in most cases:

```python
# ❌ May need 1000 epochs to converge
trainer = Trainer(model, loss, optimizer="sgd", lr=0.01)

# ✅ Often converges in 100-200 epochs
trainer = Trainer(model, loss, optimizer="adam", lr=0.001)
```

---

## 6. Avoid Unnecessary Expression Copies

```python
from neurogebra import MathForge

forge = MathForge()

# ❌ Creates new Expression object each time
for i in range(100):
    relu = forge.get("relu")        # Unnecessary repeated lookup
    result = relu.eval(x=float(i))

# ✅ Get once, use many times
relu = forge.get("relu")
for i in range(100):
    result = relu.eval(x=float(i))

# ✅✅ Even better: vectorized
relu = forge.get("relu")
x = np.arange(100, dtype=float)
results = relu.eval(x=x)
```

---

## 7. Memory Management with Autograd

When training manually with `Value`/`Tensor`, always zero gradients:

```python
from neurogebra.core.autograd import Value

x = Value(2.0)

for epoch in range(1000):
    y = x**2 + 3*x
    y.backward()
    
    x.data -= 0.01 * x.grad
    x.grad = 0.0      # ← CRITICAL: prevents gradient accumulation
```

---

## 8. Profile Your Code

Find bottlenecks with Python's built-in profiler:

```python
import time

start = time.time()

# Your Neurogebra code here
forge = MathForge()
sigmoid = forge.get("sigmoid")
x = np.linspace(-5, 5, 10000)
result = sigmoid.eval(x=x)

end = time.time()
print(f"Time: {end - start:.4f} seconds")
```

---

## Performance Cheat Sheet

| Tip | Impact | Effort |
|-----|--------|--------|
| Use NumPy arrays | 🚀🚀🚀 | Low |
| Batch evaluation | 🚀🚀🚀 | Low |
| Pre-compute gradients | 🚀🚀 | Low |
| Use Adam optimizer | 🚀🚀 | Low |
| Appropriate batch size | 🚀 | Low |
| Cache expression lookups | 🚀 | Low |
| Zero gradients | 🚀 (correctness) | Low |
| Profile bottlenecks | 🚀🚀 | Medium |

---

**Next:** [Project 1: Linear Regression →](../projects/project1-linear-regression.md)
