# Gradients & Differentiation

**Gradients are how neural networks learn.** This is the single most important concept in all of ML.

---

## The Big Picture

Training a neural network is just this:

1. Make a prediction
2. Measure the error (loss)
3. **Compute gradients** — how much does each parameter affect the error?
4. Adjust parameters in the direction that reduces error
5. Repeat

**Step 3 is what this page is about.**

---

## What is a Gradient?

A gradient (derivative) tells you: **if I change this input a tiny bit, how much does the output change?**

$$f(x) = x^2 \quad \Rightarrow \quad f'(x) = 2x$$

At $x = 3$: $f'(3) = 6$, meaning the output changes ~6 times faster than the input at that point.

---

## Symbolic Gradients in Neurogebra

Neurogebra computes gradients **symbolically** — you get the actual derivative formula, not just a number:

```python
from neurogebra import Expression

# Define a function
f = Expression("quadratic", "x**2 + 3*x + 2")

# Compute its derivative
f_prime = f.gradient("x")

# See the formula
print(f"f(x) = {f.symbolic_expr}")       # x**2 + 3*x + 2
print(f"f'(x) = {f_prime.symbolic_expr}")  # 2*x + 3

# Evaluate at specific points
print(f"f'(0) = {f_prime.eval(x=0)}")    # 3
print(f"f'(1) = {f_prime.eval(x=1)}")    # 5
print(f"f'(-1) = {f_prime.eval(x=-1)}")  # 1
```

---

## Gradients of Activation Functions

```python
from neurogebra import MathForge

forge = MathForge()

# ReLU gradient
relu = forge.get("relu")
relu_grad = relu.gradient("x")
print(f"ReLU'(x) = {relu_grad.symbolic_expr}")
# Derivative is 1 for x > 0, 0 for x < 0

# Sigmoid gradient
sigmoid = forge.get("sigmoid")
sig_grad = sigmoid.gradient("x")
print(f"Sigmoid'(x) = {sig_grad.symbolic_expr}")
# σ'(x) = σ(x) · (1 - σ(x))

# Tanh gradient
tanh = forge.get("tanh")
tanh_grad = tanh.gradient("x")
print(f"Tanh'(x) = {tanh_grad.symbolic_expr}")
# tanh'(x) = 1 - tanh²(x)
```

---

## Gradients of Loss Functions

This is directly used in training:

```python
mse = forge.get("mse")
print(f"MSE = {mse.symbolic_expr}")
# (y_pred - y_true)**2

mse_grad = mse.gradient("y_pred")
print(f"dMSE/d(y_pred) = {mse_grad.symbolic_expr}")
# 2*(y_pred - y_true)

# Interpretation:
# If prediction > target → gradient is positive → decrease prediction
# If prediction < target → gradient is negative → increase prediction
print(f"  pred=7, true=5: grad = {mse_grad.eval(y_pred=7, y_true=5)}")  # 4
print(f"  pred=3, true=5: grad = {mse_grad.eval(y_pred=3, y_true=5)}")  # -4
print(f"  pred=5, true=5: grad = {mse_grad.eval(y_pred=5, y_true=5)}")  # 0 (perfect!)
```

---

## Higher-Order Derivatives

You can differentiate multiple times:

```python
f = Expression("cubic", "x**3")

f1 = f.gradient("x")     # First derivative
f2 = f1.gradient("x")    # Second derivative
f3 = f2.gradient("x")    # Third derivative

print(f"f(x)    = {f.symbolic_expr}")    # x³
print(f"f'(x)   = {f1.symbolic_expr}")   # 3x²
print(f"f''(x)  = {f2.symbolic_expr}")   # 6x
print(f"f'''(x) = {f3.symbolic_expr}")   # 6
```

---

## Partial Derivatives

For functions with multiple variables:

```python
# f(x, y) = x²y + xy²
f = Expression("multi", "x**2*y + x*y**2")

# Partial derivative with respect to x
df_dx = f.gradient("x")
print(f"∂f/∂x = {df_dx.symbolic_expr}")  # 2*x*y + y²

# Partial derivative with respect to y
df_dy = f.gradient("y")
print(f"∂f/∂y = {df_dy.symbolic_expr}")  # x² + 2*x*y
```

---

## Gradient Descent — Using Gradients to Learn

Here's a manual gradient descent implementation to see how gradients drive learning:

```python
import numpy as np
from neurogebra import Expression

# Goal: Find x that minimizes f(x) = (x - 3)²
# Answer should be x = 3

f = Expression("parabola", "(x - 3)**2")
f_grad = f.gradient("x")  # f'(x) = 2*(x-3)

# Gradient descent
x = 10.0          # Start far from minimum
lr = 0.1          # Learning rate

print(f"{'Step':>4} | {'x':>8} | {'f(x)':>8} | {'gradient':>8}")
print("-" * 45)

for step in range(15):
    fx = f.eval(x=x)
    gx = f_grad.eval(x=x)
    
    print(f"{step:>4} | {x:>8.4f} | {fx:>8.4f} | {gx:>8.4f}")
    
    # Update: move opposite to gradient
    x = x - lr * gx

print(f"\nFinal x = {x:.4f}")  # Should be close to 3.0
```

---

## The Vanishing Gradient Problem

Some activations have gradients that become very small, making learning slow or impossible:

```python
import numpy as np

forge = MathForge()

# Sigmoid gradient becomes tiny for large |x|
sig_grad = forge.get("sigmoid").gradient("x")
for x_val in [-10, -5, 0, 5, 10]:
    g = sig_grad.eval(x=x_val)
    print(f"  sigmoid'({x_val:>3}) = {g:.6f}")

# Output:
# sigmoid'(-10) = 0.000045  ← Almost zero! Learning stops.
# sigmoid'( -5) = 0.006648
# sigmoid'(  0) = 0.250000  ← OK
# sigmoid'(  5) = 0.006648
# sigmoid'( 10) = 0.000045  ← Almost zero!

print("\nReLU doesn't have this problem:")
relu_grad = forge.get("relu").gradient("x")
for x_val in [-10, -5, 0, 5, 10]:
    g = relu_grad.eval(x=x_val)
    print(f"  relu'({x_val:>3}) = {g}")
```

!!! warning "Why this matters"
    The vanishing gradient problem is why **ReLU replaced Sigmoid** as the default activation for hidden layers. With Sigmoid, deep networks couldn't learn because gradients became too small.

---

## Numerical Gradients (Finite Differences)

Sometimes you want to verify symbolic gradients using numerical approximation:

```python
def numerical_gradient(expr, var_name, point, epsilon=1e-5):
    """Compute gradient numerically using finite differences."""
    kwargs_plus = {var_name: point + epsilon}
    kwargs_minus = {var_name: point - epsilon}
    return (expr.eval(**kwargs_plus) - expr.eval(**kwargs_minus)) / (2 * epsilon)

f = Expression("test", "x**3 + 2*x + 1")
f_grad = f.gradient("x")

x_point = 2.0
symbolic = f_grad.eval(x=x_point)
numerical = numerical_gradient(f, "x", x_point)

print(f"Symbolic gradient:  {symbolic:.6f}")   # 14.000000
print(f"Numerical gradient: {numerical:.6f}")  # 14.000000
print(f"Difference: {abs(symbolic - numerical):.10f}")  # ~0
```

---

## Summary

| Concept | What It Is | Why It Matters |
|---------|-----------|----------------|
| Gradient | Rate of change | Tells model how to adjust |
| Positive gradient | Output increases with input | Decrease the parameter |
| Negative gradient | Output decreases with input | Increase the parameter |
| Zero gradient | At a minimum/maximum | Training converged (maybe) |
| Vanishing gradient | Gradient too small | Model stops learning |
| Exploding gradient | Gradient too large | Training becomes unstable |

---

**Next:** [Expression Composition →](composition.md)
