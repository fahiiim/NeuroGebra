# Activation Functions

Activation functions are the **secret sauce** that gives neural networks the power to learn complex patterns.

---

## Why Activation Functions?

Without activation functions, a neural network is just a chain of linear transformations — which collapses into a single linear function. Activation functions add **non-linearity**, allowing networks to learn curves, boundaries, and complex relationships.

```
Without activation: Layer1(Layer2(x)) = W1*(W2*x + b2) + b1 = W*x + b  (still linear!)
With activation:    relu(W1 * relu(W2*x + b2) + b1)  → Can learn ANY function!
```

---

## Exploring Activations in Neurogebra

```python
from neurogebra import MathForge

forge = MathForge()

# List all activation functions
activations = forge.list_all(category="activation")
print(activations)
# ['relu', 'sigmoid', 'tanh', 'leaky_relu', 'swish', 'gelu', 'softplus', ...]
```

---

## ReLU (Rectified Linear Unit)

**The most popular activation function in deep learning.**

$$f(x) = \max(0, x)$$

```python
relu = forge.get("relu")

# Formula
print(relu.symbolic_expr)  # Max(0, x)

# Behavior
print(relu.eval(x=5))    # 5   (positive → pass through)
print(relu.eval(x=-3))   # 0   (negative → blocked)
print(relu.eval(x=0))    # 0   (zero → zero)

# Metadata
print(relu.metadata["description"])
# "Rectified Linear Unit - outputs x if x > 0, else 0"
print(relu.metadata["pros"])
# ['Fast computation', 'No vanishing gradient for positive values']
print(relu.metadata["cons"])
# ['Dead neurons for negative values', 'Not zero-centered']
```

!!! info "When to use ReLU"
    **Default choice for hidden layers.** Use it unless you have a specific reason not to.

---

## Sigmoid

**Maps any real number to a value between 0 and 1.**

$$f(x) = \frac{1}{1 + e^{-x}}$$

```python
sigmoid = forge.get("sigmoid")

print(sigmoid.eval(x=0))      # 0.5  (center)
print(sigmoid.eval(x=10))     # ≈ 1.0 (large positive → 1)
print(sigmoid.eval(x=-10))    # ≈ 0.0 (large negative → 0)
```

!!! info "When to use Sigmoid"
    **Output layer for binary classification** (yes/no, spam/not-spam). Outputs a probability.

---

## Tanh (Hyperbolic Tangent)

**Like Sigmoid, but outputs between -1 and 1.**

$$f(x) = \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

```python
tanh = forge.get("tanh")

print(tanh.eval(x=0))     # 0.0  (zero-centered!)
print(tanh.eval(x=5))     # ≈ 1.0
print(tanh.eval(x=-5))    # ≈ -1.0
```

!!! info "When to use Tanh"
    **Hidden layers when zero-centered output matters** (e.g., RNNs, LSTMs).

---

## Leaky ReLU

**ReLU but allows a small gradient for negative values.**

$$f(x) = \max(\alpha x, x) \quad \text{where } \alpha = 0.01$$

```python
leaky = forge.get("leaky_relu")

print(leaky.eval(x=5))    # 5
print(leaky.eval(x=-5))   # -0.05  (not zero! Small leak)

# Custom alpha
leaky02 = forge.get("leaky_relu", params={"alpha": 0.2})
print(leaky02.eval(x=-5))  # -1.0 (bigger leak)
```

!!! info "When to use Leaky ReLU"
    **When you're worried about "dead neurons"** (neurons that always output 0 with standard ReLU).

---

## Swish (SiLU)

**Self-gated activation — often outperforms ReLU.**

$$f(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

```python
swish = forge.get("swish")

print(swish.eval(x=5))    # ≈ 4.97
print(swish.eval(x=-5))   # ≈ -0.03
print(swish.eval(x=0))    # 0.0
```

!!! info "When to use Swish"
    **Modern deep networks.** Found by Google's AutoML to outperform ReLU in many cases.

---

## GELU (Gaussian Error Linear Unit)

**The activation used in GPT, BERT, and modern Transformers.**

$$f(x) = 0.5x\left(1 + \tanh\left(\sqrt{\frac{2}{\pi}}\left(x + 0.044715x^3\right)\right)\right)$$

```python
gelu = forge.get("gelu")

print(gelu.eval(x=1))     # ≈ 0.841
print(gelu.eval(x=-1))    # ≈ -0.159
print(gelu.eval(x=0))     # 0.0
```

!!! info "When to use GELU"
    **Transformer models** (BERT, GPT, ViT). The current state-of-the-art choice.

---

## Softplus

**A smooth approximation of ReLU.**

$$f(x) = \ln(1 + e^x)$$

```python
softplus = forge.get("softplus")

print(softplus.eval(x=5))    # ≈ 5.007
print(softplus.eval(x=-5))   # ≈ 0.007
print(softplus.eval(x=0))    # ≈ 0.693
```

---

## Comparison Table

| Activation | Formula | Range | Best For |
|-----------|---------|-------|----------|
| ReLU | $\max(0, x)$ | $[0, \infty)$ | Default for hidden layers |
| Sigmoid | $\frac{1}{1+e^{-x}}$ | $(0, 1)$ | Binary classification output |
| Tanh | $\tanh(x)$ | $(-1, 1)$ | RNNs, zero-centered needs |
| Leaky ReLU | $\max(\alpha x, x)$ | $(-\infty, \infty)$ | Preventing dead neurons |
| Swish | $x \cdot \sigma(x)$ | $(-\infty, \infty)$ | Modern deep networks |
| GELU | see above | $(-\infty, \infty)$ | Transformers (BERT, GPT) |
| Softplus | $\ln(1+e^x)$ | $(0, \infty)$ | Smooth ReLU alternative |

---

## Side-by-Side Comparison

```python
from neurogebra import MathForge
import numpy as np

forge = MathForge()

x = 1.0
print(f"{'Activation':>12} | f({x})")
print("-" * 30)

for name in ["relu", "sigmoid", "tanh", "leaky_relu", "swish", "gelu", "softplus"]:
    expr = forge.get(name)
    val = expr.eval(x=x)
    print(f"{name:>12} | {val:.4f}")
```

Output:
```
  Activation | f(1.0)
------------------------------
        relu | 1.0000
     sigmoid | 0.7311
        tanh | 0.7616
  leaky_relu | 1.0000
       swish | 0.7311
        gelu | 0.8412
    softplus | 1.3133
```

---

## Gradients of Activation Functions

Understanding **how gradients flow** through activation functions is crucial:

```python
for name in ["relu", "sigmoid", "tanh", "swish"]:
    expr = forge.get(name)
    grad = expr.gradient("x")
    
    print(f"{name}:")
    print(f"  f(x) = {expr.symbolic_expr}")
    print(f"  f'(x) = {grad.symbolic_expr}")
    print(f"  f'(1.0) = {grad.eval(x=1.0):.4f}")
    print()
```

---

## Try It Yourself!

!!! example "Exercise"
    1. Get all activation functions and evaluate them at `x = -2, -1, 0, 1, 2`
    2. Which activation outputs the largest value at `x = 1`?
    3. Which activation has the largest gradient at `x = 0`?
    4. Create a custom activation: `f(x) = x * sigmoid(2*x)`

---

**Next:** [Loss Functions →](losses.md)
