# Python Basics Recap

If you know Python, this is a quick refresher of the concepts you'll need for Machine Learning. If something looks new, take a moment to understand it.

---

## Variables and Data Types

```python
# Numbers
age = 25              # int
temperature = 98.6    # float
learning_rate = 0.01  # float (very common in ML!)

# Strings
name = "neural network"

# Booleans
is_training = True
converged = False
```

!!! tip "In ML, most values are `float`"
    Weights, biases, learning rates, loss values — almost everything in ML is a floating-point number.

---

## Lists — Ordered Collections

```python
# A list of numbers (like a dataset)
scores = [85, 92, 78, 95, 88]

# Access elements (0-indexed)
print(scores[0])   # 85 (first)
print(scores[-1])  # 88 (last)

# List length
print(len(scores))  # 5

# Slicing
print(scores[1:3])  # [92, 78]

# List comprehension (very common in ML code)
doubled = [x * 2 for x in scores]
print(doubled)  # [170, 184, 156, 190, 176]
```

---

## Dictionaries — Key-Value Pairs

```python
# Model parameters
params = {
    "learning_rate": 0.01,
    "epochs": 100,
    "batch_size": 32
}

# Access
print(params["learning_rate"])  # 0.01

# Update
params["epochs"] = 200

# Loop through
for key, value in params.items():
    print(f"{key} = {value}")
```

!!! note "Why dictionaries matter in ML"
    Model hyperparameters, configuration settings, and expression parameters are almost always stored as dictionaries.

---

## Functions

```python
# Basic function
def relu(x):
    """ReLU activation function."""
    if x > 0:
        return x
    else:
        return 0

print(relu(5))    # 5
print(relu(-3))   # 0

# Function with default parameters
def train(epochs=100, lr=0.01):
    print(f"Training for {epochs} epochs with lr={lr}")

train()                    # Uses defaults
train(epochs=200, lr=0.001)  # Custom values
```

---

## Loops

```python
# For loop — iterating over data
data = [1, 2, 3, 4, 5]
total = 0
for value in data:
    total += value
print(f"Sum = {total}")  # 15

# Range-based loop — for epochs
for epoch in range(5):
    print(f"Epoch {epoch}")
# Epoch 0, 1, 2, 3, 4

# While loop — until convergence
loss = 10.0
while loss > 0.1:
    loss = loss * 0.5  # Simulating loss decreasing
    print(f"Loss: {loss:.2f}")
```

---

## Classes and Objects

Understanding classes is essential because Neurogebra uses them everywhere.

```python
class Neuron:
    """A simple neuron."""
    
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias
    
    def forward(self, x):
        """Compute output: weight * x + bias."""
        return self.weight * x + self.bias
    
    def __repr__(self):
        return f"Neuron(w={self.weight}, b={self.bias})"

# Create a neuron
n = Neuron(weight=2.0, bias=1.0)

# Use it
print(n.forward(3))    # 2.0 * 3 + 1.0 = 7.0
print(n)               # Neuron(w=2.0, b=1.0)
```

!!! info "Neurogebra Example"
    In Neurogebra, `Expression` is a class. When you do `relu = forge.get("relu")`, you get an `Expression` object with methods like `.eval()`, `.gradient()`, and `.explain()`.

---

## Lambda Functions

Short, one-line functions used frequently in ML:

```python
# Lambda: anonymous function
square = lambda x: x ** 2
print(square(4))  # 16

# Common in sorting and filtering
activations = ["relu", "sigmoid", "tanh", "gelu"]
sorted_acts = sorted(activations, key=lambda a: len(a))
print(sorted_acts)  # ['relu', 'tanh', 'gelu', 'sigmoid']
```

---

## Error Handling

```python
try:
    result = 10 / 0
except ZeroDivisionError:
    print("Cannot divide by zero!")

# In ML context
try:
    from neurogebra import MathForge
    forge = MathForge()
    expr = forge.get("nonexistent_function")
except KeyError as e:
    print(f"Expression not found: {e}")
```

---

## F-Strings (Formatted Strings)

Used constantly for printing training progress:

```python
epoch = 42
loss = 0.0234
accuracy = 0.9567

print(f"Epoch {epoch}: Loss = {loss:.4f}, Accuracy = {accuracy:.2%}")
# Output: Epoch 42: Loss = 0.0234, Accuracy = 95.67%
```

The formatting codes:

| Code | Meaning | Example |
|------|---------|---------|
| `:.4f` | 4 decimal places | `0.0234` |
| `:.2f` | 2 decimal places | `0.02` |
| `:.2%` | Percentage | `95.67%` |
| `:.2e` | Scientific notation | `2.34e-02` |

---

## Importing Modules

```python
# Import entire module
import numpy as np

# Import specific items
from neurogebra import MathForge, Expression

# Import with alias
from neurogebra.core.trainer import Trainer
```

---

## Try It Yourself!

!!! example "Exercise"
    ```python
    # 1. Create a dictionary of hyperparameters
    config = {"lr": 0.001, "epochs": 50, "optimizer": "adam"}
    
    # 2. Write a function that simulates training
    def fake_train(config):
        for epoch in range(config["epochs"]):
            loss = 1.0 / (epoch + 1)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: loss = {loss:.4f}")
    
    fake_train(config)
    ```

---

**Next:** [NumPy Essentials →](numpy.md)
