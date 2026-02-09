# NumPy Essentials

**NumPy** is the foundation of all numerical computing in Python. Every ML framework (including Neurogebra) uses NumPy under the hood.

---

## What is NumPy?

NumPy provides **fast array operations**. Instead of using slow Python loops, NumPy operates on entire arrays at once.

```python
import numpy as np
```

!!! info "Convention"
    Everyone imports NumPy as `np`. This is a universal convention.

---

## Creating Arrays

```python
import numpy as np

# From a list
a = np.array([1, 2, 3, 4, 5])
print(a)        # [1 2 3 4 5]
print(type(a))  # <class 'numpy.ndarray'>

# Array of zeros
zeros = np.zeros(5)
print(zeros)  # [0. 0. 0. 0. 0.]

# Array of ones
ones = np.ones(5)
print(ones)  # [1. 1. 1. 1. 1.]

# Range of values
x = np.arange(0, 10, 2)  # start, stop, step
print(x)  # [0 2 4 6 8]

# Evenly spaced values (very common for plotting)
x = np.linspace(0, 1, 5)  # start, stop, num_points
print(x)  # [0.   0.25 0.5  0.75 1.  ]

# Random values
random_data = np.random.randn(5)  # 5 random numbers from normal distribution
print(random_data)
```

---

## Array Properties

```python
a = np.array([[1, 2, 3],
              [4, 5, 6]])

print(a.shape)   # (2, 3) — 2 rows, 3 columns
print(a.ndim)    # 2 — dimensions
print(a.size)    # 6 — total elements
print(a.dtype)   # int64 — data type
```

!!! tip "Shapes in ML"
    In ML, you'll constantly deal with shapes:
    
    - `(100,)` — 100 data points (1D)
    - `(100, 10)` — 100 samples, 10 features (2D)
    - `(32, 28, 28)` — 32 images of 28×28 pixels (3D)

---

## Array Operations (Element-wise)

NumPy operations work on **every element** simultaneously:

```python
a = np.array([1, 2, 3, 4, 5])

# Arithmetic
print(a + 10)     # [11 12 13 14 15]
print(a * 2)      # [ 2  4  6  8 10]
print(a ** 2)     # [ 1  4  9 16 25]

# Between arrays
b = np.array([10, 20, 30, 40, 50])
print(a + b)      # [11 22 33 44 55]
print(a * b)      # [ 10  40  90 160 250]
```

**Why this matters:** In ML, we compute things like `predictions - targets` on thousands of data points at once.

---

## Mathematical Functions

```python
x = np.array([-2, -1, 0, 1, 2])

# Common functions
print(np.abs(x))      # [2 1 0 1 2]
print(np.exp(x))      # [0.135 0.368 1.000 2.718 7.389]
print(np.log(np.abs(x) + 1))  # logarithm (add 1 to avoid log(0))
print(np.sqrt(np.abs(x)))     # [1.414 1.000 0.000 1.000 1.414]

# Implement ReLU with NumPy
print(np.maximum(0, x))  # [0 0 0 1 2] — ReLU!

# Implement Sigmoid with NumPy
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

print(sigmoid(x))  # [0.119 0.269 0.5 0.731 0.881]
```

---

## Statistical Functions

```python
data = np.array([85, 92, 78, 95, 88, 76, 91])

print(np.mean(data))    # 86.43 — average
print(np.std(data))     # 6.90  — standard deviation
print(np.min(data))     # 76
print(np.max(data))     # 95
print(np.sum(data))     # 605
print(np.median(data))  # 88.0
```

---

## Reshaping Arrays

```python
a = np.array([1, 2, 3, 4, 5, 6])

# Reshape to 2x3
b = a.reshape(2, 3)
print(b)
# [[1 2 3]
#  [4 5 6]]

# Reshape to column vector
c = a.reshape(-1, 1)  # -1 means "figure it out"
print(c.shape)  # (6, 1)

# Flatten back to 1D
d = b.flatten()
print(d)  # [1 2 3 4 5 6]
```

---

## Indexing and Slicing

```python
a = np.array([10, 20, 30, 40, 50])

# Basic indexing
print(a[0])     # 10
print(a[-1])    # 50

# Slicing
print(a[1:4])   # [20 30 40]
print(a[:3])    # [10 20 30]

# Boolean indexing (filtering)
print(a[a > 25])   # [30 40 50]
print(a[a % 20 == 0])  # [20 40]

# 2D indexing
matrix = np.array([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]])
print(matrix[0, 0])    # 1 (row 0, col 0)
print(matrix[1, :])    # [4 5 6] (entire row 1)
print(matrix[:, 2])    # [3 6 9] (entire column 2)
```

---

## The Dot Product

The dot product is the most important operation in neural networks:

```python
# Vectors
weights = np.array([0.5, -0.3, 0.8])
inputs = np.array([1.0, 2.0, 3.0])

# Dot product: sum of element-wise multiplication
result = np.dot(weights, inputs)
print(result)  # 0.5*1 + (-0.3)*2 + 0.8*3 = 1.5

# This is exactly what a neuron does!
bias = 0.1
output = np.dot(weights, inputs) + bias
print(output)  # 1.6
```

!!! note "Neuron = Dot Product + Bias + Activation"
    Every neuron in a neural network computes: $output = activation(weights \cdot inputs + bias)$

---

## Generating Synthetic Data

This is very useful for testing ML models:

```python
import numpy as np

# Linear data: y = 2x + 1 + noise
np.random.seed(42)  # For reproducibility
X = np.linspace(0, 10, 100)
noise = np.random.normal(0, 0.5, 100)
y = 2 * X + 1 + noise

print(f"X shape: {X.shape}")  # (100,)
print(f"y shape: {y.shape}")  # (100,)
print(f"X range: [{X.min():.1f}, {X.max():.1f}]")
print(f"y range: [{y.min():.1f}, {y.max():.1f}]")
```

---

## NumPy in Neurogebra

Neurogebra expressions accept NumPy arrays:

```python
from neurogebra import MathForge
import numpy as np

forge = MathForge()
relu = forge.get("relu")

# Evaluate on an entire array at once!
x = np.array([-3, -1, 0, 1, 3])
result = relu.eval(x=x)
print(result)  # [0 0 0 1 3]
```

---

## Try It Yourself!

!!! example "Exercise"
    ```python
    import numpy as np
    
    # 1. Create a dataset: y = 3x² - 2x + 1
    X = np.linspace(-5, 5, 50)
    y = 3 * X**2 - 2 * X + 1
    
    # 2. Compute statistics
    print(f"Mean of y: {np.mean(y):.2f}")
    print(f"Std of y: {np.std(y):.2f}")
    
    # 3. Implement sigmoid using NumPy
    sigmoid = 1 / (1 + np.exp(-X))
    print(f"Sigmoid range: [{sigmoid.min():.3f}, {sigmoid.max():.3f}]")
    ```

---

**Next:** [Data Handling with Python →](data-handling.md)
