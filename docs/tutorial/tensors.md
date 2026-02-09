# Tensors

Tensors extend the `Value` concept to **multi-dimensional arrays** — enabling batched operations essential for efficient ML.

---

## What is a Tensor?

A tensor is a multi-dimensional array with **automatic gradient tracking**:

- Scalar (0D tensor): `42`
- Vector (1D tensor): `[1, 2, 3]`
- Matrix (2D tensor): `[[1, 2], [3, 4]]`
- 3D tensor: A batch of matrices

---

## Creating Tensors

```python
from neurogebra.core.autograd import Tensor

# From a list
t = Tensor([1.0, 2.0, 3.0])
print(t)        # Tensor(shape=(3,), requires_grad=False)
print(t.data)   # [1. 2. 3.]

# With gradient tracking enabled
t = Tensor([1.0, 2.0, 3.0], requires_grad=True)
print(t)        # Tensor(shape=(3,), requires_grad=True)
print(t.grad)   # [0. 0. 0.] — initialized to zeros

# 2D tensor
m = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
print(m.shape)  # (2, 2)
print(m.ndim)   # 2
```

---

## Tensor Properties

```python
t = Tensor([1.0, 2.0, 3.0], requires_grad=True)

print(t.shape)          # (3,)
print(t.ndim)           # 1
print(t.data)           # [1. 2. 3.]
print(t.grad)           # [0. 0. 0.]
print(t.requires_grad)  # True
```

---

## Tensor Operations

### Element-wise Arithmetic

```python
a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
b = Tensor([4.0, 5.0, 6.0], requires_grad=True)

c = a + b              # [5.0, 7.0, 9.0]
d = a * b              # [4.0, 10.0, 18.0]
e = a - b              # [-3.0, -3.0, -3.0]
f = a ** 2             # [1.0, 4.0, 9.0]

print(f"a + b = {c.data}")
print(f"a * b = {d.data}")
print(f"a ** 2 = {f.data}")
```

### Scalar Operations

```python
a = Tensor([1.0, 2.0, 3.0], requires_grad=True)

b = a * 3              # [3.0, 6.0, 9.0]
c = a + 10             # [11.0, 12.0, 13.0]
```

### Reduction Operations

```python
a = Tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)

s = a.sum()     # Sum of all elements → 10.0
m = a.mean()    # Mean of all elements → 2.5

print(f"sum = {s.data}")   # 10.0
print(f"mean = {m.data}")  # 2.5
```

---

## Backward Pass with Tensors

```python
from neurogebra.core.autograd import Tensor

# Create tensor with gradient tracking
x = Tensor([1.0, 2.0, 3.0], requires_grad=True)

# Computation: y = sum(x²) = 1 + 4 + 9 = 14
y = (x ** 2).sum()

# Backward pass
y.backward()

print(f"x = {x.data}")       # [1. 2. 3.]
print(f"y = x² sum = {y.data}")  # 14.0
print(f"dy/dx = {x.grad}")   # [2. 4. 6.] — which is 2*x
```

---

## Practical Example: Batch MSE Loss

```python
from neurogebra.core.autograd import Tensor

# Predictions and targets (batch of 4)
predictions = Tensor([2.5, 0.1, 3.0, 7.0], requires_grad=True)
targets = Tensor([3.0, 0.0, 2.5, 8.0])

# MSE = mean((pred - target)²)
diff = predictions - targets
squared = diff ** 2
loss = squared.mean()

print(f"Predictions: {predictions.data}")
print(f"Targets:     {targets.data}")
print(f"MSE Loss:    {loss.data:.4f}")

# Backpropagate
loss.backward()
print(f"Gradients:   {predictions.grad}")
# Each gradient = 2*(pred-target)/n
```

---

## Zeroing Gradients

Always zero gradients between training iterations:

```python
x = Tensor([1.0, 2.0], requires_grad=True)

# First iteration
y = (x ** 2).sum()
y.backward()
print(f"After 1st backward: grad = {x.grad}")  # [2. 4.]

# Zero and compute again
x.zero_grad()
y = (x ** 2).sum()
y.backward()
print(f"After zero + 2nd backward: grad = {x.grad}")  # [2. 4.] — correct
```

---

## Key Differences: Value vs Tensor

| Feature | Value | Tensor |
|---------|-------|--------|
| Data type | Single float | NumPy array |
| Shape | Scalar | Any shape |
| Use case | Educational, simple networks | Batched training |
| Gradient | Single float | Array (same shape as data) |
| Operations | +, *, **, relu, sigmoid, tanh | +, *, **, sum, mean |

---

**Next:** [ModelBuilder →](model-builder.md)
