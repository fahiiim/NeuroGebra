# Math Behind ML

Don't panic — the math behind ML is simpler than you think. Neurogebra was built specifically to make this math **visible and understandable**.

---

## The Three Pillars of ML Math

```
1. Linear Algebra  → How data flows through the model
2. Calculus         → How the model learns (gradients)
3. Probability      → How we measure uncertainty
```

---

## Pillar 1: Linear Algebra

### Vectors

A vector is just a list of numbers. In ML, your data is vectors.

```python
import numpy as np

# A data point with 3 features
# [height, weight, age]
person = np.array([170, 65, 25])

# Weights of a neuron
weights = np.array([0.5, -0.3, 0.1])
```

### The Dot Product

The most important operation in neural networks:

$$\mathbf{w} \cdot \mathbf{x} = w_1 x_1 + w_2 x_2 + w_3 x_3$$

```python
# This is what EVERY neuron computes
output = np.dot(weights, person)
# = 0.5*170 + (-0.3)*65 + 0.1*25
# = 85 - 19.5 + 2.5
# = 68.0
```

### Matrices

A matrix is a 2D array. A neural network layer is a matrix multiplication:

$$\mathbf{y} = \mathbf{W} \cdot \mathbf{x} + \mathbf{b}$$

```python
# Layer with 3 inputs, 2 outputs
W = np.array([[0.5, -0.3, 0.1],
              [0.2,  0.4, -0.1]])  # Shape: (2, 3)
b = np.array([0.1, -0.2])          # Shape: (2,)
x = np.array([170, 65, 25])        # Shape: (3,)

output = W @ x + b  # @ is matrix multiplication
print(output)  # Shape: (2,) — two output neurons
```

Neurogebra has linear algebra expressions:

```python
from neurogebra import MathForge

forge = MathForge()
expressions = forge.list_all(category="linalg")
print(expressions)
```

---

## Pillar 2: Calculus (Derivatives & Gradients)

### What is a Derivative?

A derivative tells you how much the output changes when you change the input **a tiny bit**.

$$f(x) = x^2 \quad \Rightarrow \quad f'(x) = 2x$$

At $x = 3$: the derivative is $2 \times 3 = 6$. This means: if you increase $x$ by a tiny amount, $f(x)$ increases by about 6 times that amount.

```python
from neurogebra import Expression

f = Expression("quadratic", "x**2")
f_prime = f.gradient("x")

print(f"f(3) = {f.eval(x=3)}")           # 9
print(f"f'(3) = {f_prime.eval(x=3)}")    # 6
```

### Why Derivatives Matter in ML

The **loss function** measures error. The **derivative of the loss with respect to each weight** tells us how to adjust that weight:

```python
from neurogebra import MathForge

forge = MathForge()

# MSE loss: L = (prediction - target)²
mse = forge.get("mse")
print(mse.symbolic_expr)  # (y_pred - y_true)**2

# Gradient: dL/d(prediction) = 2*(prediction - target)
grad = mse.gradient("y_pred")
print(grad.symbolic_expr)  # 2*(y_pred - y_true)

# If prediction=5, target=3:
# gradient = 2*(5-3) = 4 (positive → decrease prediction)
print(grad.eval(y_pred=5, y_true=3))  # 4.0
```

### The Chain Rule

For composite functions, derivatives multiply:

$$\frac{d}{dx} f(g(x)) = f'(g(x)) \cdot g'(x)$$

This is exactly what **backpropagation** does — it applies the chain rule through every layer of a neural network.

```python
from neurogebra.core.autograd import Value

# Chain of operations
x = Value(2.0)
h = x ** 2        # h = x² = 4
y = h + 3         # y = h + 3 = 7
z = y * 2         # z = y * 2 = 14

# Backprop automatically applies chain rule
z.backward()

print(f"dz/dx = {x.grad}")  # dz/dh * dh/dx = 2 * 2x = 2*4 = 8
```

### Gradient Descent

The core learning algorithm: move in the **opposite direction** of the gradient:

$$w_{new} = w_{old} - \alpha \cdot \frac{\partial L}{\partial w}$$

Where $\alpha$ is the **learning rate**.

```python
# Manual gradient descent
w = 5.0          # Start with a guess
lr = 0.1         # Learning rate
target = 0.0     # Want to minimize w²

for step in range(20):
    loss = w ** 2           # Loss function
    gradient = 2 * w        # dL/dw = 2w
    w = w - lr * gradient   # Update step
    
    if step % 5 == 0:
        print(f"Step {step}: w = {w:.4f}, loss = {loss:.4f}")

# w gets closer and closer to 0 (the minimum of w²)
```

---

## Pillar 3: Probability & Statistics

### Mean (Average)

$$\mu = \frac{1}{n} \sum_{i=1}^{n} x_i$$

```python
import numpy as np

data = np.array([85, 92, 78, 95, 88])
print(f"Mean: {np.mean(data)}")  # 87.6
```

### Variance and Standard Deviation

How spread out the data is:

$$\sigma^2 = \frac{1}{n} \sum_{i=1}^{n} (x_i - \mu)^2$$

```python
print(f"Variance: {np.var(data):.2f}")  # 33.84
print(f"Std Dev:  {np.std(data):.2f}")  # 5.82
```

### The Normal (Gaussian) Distribution

Most data in ML follows a bell curve:

```python
# Generate normally distributed data
samples = np.random.normal(mean=0, scale=1, size=1000)
print(f"Mean: {samples.mean():.2f}")  # ≈ 0
print(f"Std:  {samples.std():.2f}")   # ≈ 1
```

### Softmax — Probabilities from Numbers

Converts raw numbers into probabilities that sum to 1:

$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

```python
def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return exp_x / exp_x.sum()

scores = np.array([2.0, 1.0, 0.1])
probs = softmax(scores)
print(probs)  # [0.659, 0.242, 0.099]
print(probs.sum())  # 1.0 — they're probabilities!
```

---

## Putting It All Together: A Neuron

A single neuron combines all three pillars:

1. **Linear Algebra:** $z = \mathbf{w} \cdot \mathbf{x} + b$ (dot product)
2. **Calculus:** Gradient descent learns $\mathbf{w}$ and $b$
3. **Probability:** Activation (sigmoid) outputs a probability

```python
from neurogebra.core.autograd import Value

# Inputs
x1, x2 = Value(1.0), Value(2.0)

# Weights & bias (the parameters to learn)
w1, w2, b = Value(0.5), Value(-0.3), Value(0.1)

# Step 1: Linear combination (Linear Algebra)
z = w1 * x1 + w2 * x2 + b  # 0.5*1 + (-0.3)*2 + 0.1 = 0.0

# Step 2: Activation (Probability - squash to 0-1)
output = z.sigmoid()  # σ(0.0) = 0.5

# Step 3: Compute gradients (Calculus)
output.backward()

print(f"Output: {output.data:.4f}")    # 0.5000
print(f"dout/dw1 = {w1.grad:.4f}")     # How to adjust w1
print(f"dout/dw2 = {w2.grad:.4f}")     # How to adjust w2
print(f"dout/db  = {b.grad:.4f}")      # How to adjust bias
```

---

## Neurogebra Makes Math Visible

Instead of hiding math inside C++ kernels like PyTorch:

```python
from neurogebra import MathForge

forge = MathForge()

# See the formula
sigmoid = forge.get("sigmoid")
print(sigmoid.symbolic_expr)  # 1/(1 + exp(-x))

# See the derivative
print(sigmoid.gradient("x").symbolic_expr)
# exp(-x)/(1 + exp(-x))² — which equals σ(x)·(1-σ(x))

# Explain it
print(sigmoid.explain())
```

!!! success "Key Insight"
    You don't need to memorize derivatives. Neurogebra computes them symbolically. But **understanding what they mean** is crucial for debugging and improving your models.

---

**Next:** [MathForge — The Core →](../tutorial/mathforge.md)
