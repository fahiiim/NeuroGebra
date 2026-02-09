# Your First Program

Let's write your very first Neurogebra program. By the end of this page, you'll understand the core concept of the library.

---

## The Big Idea

In Machine Learning, everything is **math**. Neural networks use:

- **Activation functions** (like ReLU, Sigmoid) to add non-linearity
- **Loss functions** (like MSE, Cross-Entropy) to measure errors
- **Gradients** (derivatives) to learn from mistakes

Neurogebra gives you all of these as **readable, explainable mathematical expressions**.

---

## Step 1: Import Neurogebra

```python
from neurogebra import MathForge
```

`MathForge` is your **main tool**. Think of it as a toolbox that contains every mathematical function used in ML.

---

## Step 2: Create a MathForge Instance

```python
forge = MathForge()
```

This loads 50+ pre-built mathematical expressions into memory.

---

## Step 3: Get an Expression

```python
relu = forge.get("relu")
```

Now `relu` is a **mathematical expression object**. It's not just a function — it's an object that knows:

- Its formula
- How to explain itself
- How to compute its gradient
- How to evaluate numerically

---

## Step 4: Explore the Expression

```python
# See the name
print(relu.name)
# Output: relu

# See the symbolic formula
print(relu.symbolic_expr)
# Output: Max(0, x)

# Get a plain-English explanation
print(relu.explain())
```

---

## Step 5: Evaluate It

```python
# Positive input → returns the input
print(relu.eval(x=5))     # Output: 5
print(relu.eval(x=2.7))   # Output: 2.7

# Negative input → returns 0
print(relu.eval(x=-3))    # Output: 0
print(relu.eval(x=-100))  # Output: 0

# Zero → returns 0
print(relu.eval(x=0))     # Output: 0
```

!!! note "What is ReLU?"
    ReLU stands for **Rectified Linear Unit**. It's the most popular activation function in deep learning. The rule is simple: if the input is positive, output it as-is; if negative, output 0.
    
    Formula: $f(x) = \max(0, x)$

---

## Step 6: Compute the Gradient

```python
relu_grad = relu.gradient("x")
print(relu_grad)
```

The gradient (derivative) tells you how the output changes when the input changes. This is the foundation of how neural networks **learn**.

---

## Step 7: Try More Expressions

```python
# Sigmoid — maps any number to (0, 1)
sigmoid = forge.get("sigmoid")
print(sigmoid.eval(x=0))     # 0.5
print(sigmoid.eval(x=10))    # ≈ 1.0
print(sigmoid.eval(x=-10))   # ≈ 0.0

# Mean Squared Error — measures prediction error
mse = forge.get("mse")
print(mse.eval(y_pred=3.0, y_true=5.0))  # 4.0 (error = (3-5)² = 4)
```

---

## Complete First Program

Here's everything together:

```python
from neurogebra import MathForge

# Create the forge
forge = MathForge()

# === ACTIVATION FUNCTIONS ===
relu = forge.get("relu")
sigmoid = forge.get("sigmoid")

# Evaluate
print("ReLU(5)  =", relu.eval(x=5))        # 5
print("ReLU(-3) =", relu.eval(x=-3))       # 0
print("Sigmoid(0) =", sigmoid.eval(x=0))   # 0.5

# Explain
print("\n--- What is ReLU? ---")
print(relu.explain())

# Gradient
relu_grad = relu.gradient("x")
print("\nReLU gradient:", relu_grad)

# === LOSS FUNCTIONS ===
mse = forge.get("mse")
error = mse.eval(y_pred=2.5, y_true=3.0)
print(f"\nMSE(pred=2.5, true=3.0) = {error}")

# === EXPLORE ===
print("\nAll available expressions:")
print(forge.list_all())

print("\nActivation functions:")
print(forge.list_all(category="activation"))
```

---

## Try It Yourself!

!!! example "Exercise"
    1. Get the `tanh` activation and evaluate it at `x=0`, `x=1`, `x=-1`
    2. Get the `mae` (Mean Absolute Error) loss and evaluate it
    3. Use `forge.search("smooth")` to find smooth activation functions
    4. Call `.explain()` on any expression that interests you

---

## Key Takeaways

| Concept | Code | What It Does |
|---------|------|--------------|
| Create forge | `forge = MathForge()` | Loads all math expressions |
| Get expression | `forge.get("relu")` | Gets a specific expression |
| Evaluate | `expr.eval(x=5)` | Computes the result |
| Explain | `expr.explain()` | Plain-English explanation |
| Gradient | `expr.gradient("x")` | Computes the derivative |
| List all | `forge.list_all()` | Shows everything available |
| Search | `forge.search("query")` | Finds matching expressions |

---

**Next:** [How Neurogebra Works →](how-it-works.md)
