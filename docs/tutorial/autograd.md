# Autograd Engine

Neurogebra includes a built-in **automatic differentiation engine** that tracks computations and computes gradients automatically — just like PyTorch's autograd, but simpler and more educational.

---

## What is Autograd?

**Autograd** = **Auto**matic **Grad**ient computation.

Instead of computing derivatives by hand or symbolically, autograd:

1. Records every operation you perform (builds a computation graph)
2. When you call `.backward()`, it walks backwards through the graph
3. Computes the gradient of each value using the chain rule

---

## The Value Class

`Value` wraps a number and tracks its gradient:

```python
from neurogebra.core.autograd import Value

# Create values
a = Value(2.0)
b = Value(3.0)

print(a)  # Value(data=2.0, grad=0.0)
print(b)  # Value(data=3.0, grad=0.0)
```

---

## Forward Pass — Build the Graph

```python
from neurogebra.core.autograd import Value

a = Value(2.0)
b = Value(3.0)

# Each operation creates a new Value and records the connection
c = a + b      # c = 5.0
d = a * b      # d = 6.0
e = c + d      # e = 11.0

print(f"a = {a.data}")  # 2.0
print(f"b = {b.data}")  # 3.0
print(f"c = a + b = {c.data}")  # 5.0
print(f"d = a * b = {d.data}")  # 6.0
print(f"e = c + d = {e.data}")  # 11.0
```

The computation graph looks like:

```
a (2.0) ──┬──[+]── c (5.0) ──┐
           │                   [+]── e (11.0)
b (3.0) ──┼──[+]── c         │
           │                   │
a (2.0) ──┤                   │
           └──[*]── d (6.0) ──┘
b (3.0) ──┘
```

---

## Backward Pass — Compute Gradients

```python
# Compute de/da and de/db
e.backward()

print(f"de/da = {a.grad}")  # 4.0  (from + path: 1, from * path: b=3 → total: 1+3=4)
print(f"de/db = {b.grad}")  # 3.0  (from + path: 1, from * path: a=2 → total: 1+2=3)
```

Let's verify manually:

- $e = (a + b) + (a \cdot b) = a + b + ab$
- $\frac{\partial e}{\partial a} = 1 + b = 1 + 3 = 4$ ✅
- $\frac{\partial e}{\partial b} = 1 + a = 1 + 2 = 3$ ✅

---

## Supported Operations

```python
from neurogebra.core.autograd import Value

x = Value(2.0)

# Basic arithmetic
y = x + 3        # Addition
y = x * 3        # Multiplication
y = x ** 2       # Power
y = x - 1        # Subtraction
y = x / 2        # Division
y = -x           # Negation

# Activation functions
y = x.relu()     # ReLU
y = x.sigmoid()  # Sigmoid
y = x.tanh()     # Tanh
y = x.exp()      # Exponential
y = x.log()      # Natural log
```

---

## Building a Neuron

A single neuron is: $output = activation(w_1 x_1 + w_2 x_2 + b)$

```python
from neurogebra.core.autograd import Value

# Inputs
x1 = Value(2.0)
x2 = Value(3.0)

# Learnable parameters
w1 = Value(0.5)
w2 = Value(-0.3)
b = Value(0.1)

# Forward pass
z = w1 * x1 + w2 * x2 + b   # Linear: 0.5*2 + (-0.3)*3 + 0.1 = 0.2
output = z.sigmoid()          # Activation: σ(0.2) ≈ 0.5498

# Backward pass
output.backward()

print(f"Output: {output.data:.4f}")
print(f"Gradients:")
print(f"  dout/dw1 = {w1.grad:.4f}")
print(f"  dout/dw2 = {w2.grad:.4f}")
print(f"  dout/db  = {b.grad:.4f}")
```

---

## Manual Training Loop with Autograd

This is how PyTorch works internally — and now you can see every step:

```python
from neurogebra.core.autograd import Value
import random

# Training data: y = 2x + 1
data = [(1, 3), (2, 5), (3, 7), (4, 9)]

# Learnable parameters
w = Value(0.0)
b = Value(0.0)
learning_rate = 0.01

for epoch in range(100):
    total_loss = Value(0.0)
    
    for x_val, y_val in data:
        # Forward pass
        x = Value(x_val)
        y_pred = w * x + b
        
        # Loss (MSE for one sample)
        loss = (y_pred - Value(y_val)) ** 2
        total_loss = total_loss + loss
    
    # Backward pass
    total_loss.backward()
    
    # Update parameters (gradient descent)
    w.data -= learning_rate * w.grad
    b.data -= learning_rate * b.grad
    
    # Reset gradients for next epoch
    w.grad = 0.0
    b.grad = 0.0
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch:>3}: loss = {total_loss.data:.4f}, w = {w.data:.4f}, b = {b.data:.4f}")

print(f"\nLearned: y = {w.data:.2f}x + {b.data:.2f}")
# Expected: y ≈ 2.00x + 1.00
```

---

## Building a Mini Neural Network

```python
from neurogebra.core.autograd import Value
import random

class Neuron:
    def __init__(self, n_inputs):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_inputs)]
        self.b = Value(0.0)
    
    def __call__(self, x):
        # w · x + b
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        return act.relu()
    
    def parameters(self):
        return self.w + [self.b]

class MLP:
    def __init__(self, n_inputs, layer_sizes):
        sizes = [n_inputs] + layer_sizes
        self.layers = []
        for i in range(len(layer_sizes)):
            neurons = [Neuron(sizes[i]) for _ in range(sizes[i+1])]
            self.layers.append(neurons)
    
    def __call__(self, x):
        for layer in self.layers:
            x = [neuron(x) for neuron in layer]
        return x[0] if len(x) == 1 else x
    
    def parameters(self):
        return [p for layer in self.layers for neuron in layer for p in neuron.parameters()]

# Create a tiny network: 2 inputs → 4 hidden → 1 output
random.seed(42)
model = MLP(2, [4, 1])

print(f"Total parameters: {len(model.parameters())}")
# (2+1)*4 + (4+1)*1 = 12 + 5 = 17 parameters
```

---

## Zero Grad — Why It Matters

```python
x = Value(3.0)
y = x ** 2
y.backward()
print(f"After first backward: x.grad = {x.grad}")  # 6.0

# If we compute again WITHOUT zeroing:
y = x ** 2
y.backward()
print(f"After second backward: x.grad = {x.grad}")  # 12.0 — WRONG! Accumulated!

# Always zero gradients between iterations:
x.zero_grad()
y = x ** 2
y.backward()
print(f"After zero + backward: x.grad = {x.grad}")  # 6.0 — Correct!
```

---

## Summary

| Step | What Happens | Code |
|------|-------------|------|
| Create values | Wrap numbers | `x = Value(2.0)` |
| Forward pass | Compute result | `y = w * x + b` |
| Backward pass | Compute gradients | `y.backward()` |
| Read gradients | See derivatives | `w.grad` |
| Update weights | Learn | `w.data -= lr * w.grad` |
| Zero gradients | Reset for next iteration | `w.zero_grad()` |

---

**Next:** [Tensors →](tensors.md)
