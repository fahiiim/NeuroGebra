# Data Handling with Python

Before you can train any ML model, you need to **load, clean, and prepare data**. This page covers the essential techniques.

---

## Reading Data from Files

### CSV Files

```python
import numpy as np

# Using NumPy (simple numeric data)
data = np.loadtxt("data.csv", delimiter=",", skiprows=1)
print(data.shape)

# Better: Using Python's built-in csv module
import csv

with open("data.csv", "r") as f:
    reader = csv.reader(f)
    headers = next(reader)  # Skip header row
    data = [row for row in reader]
```

### JSON Files

```python
import json

with open("config.json", "r") as f:
    config = json.load(f)

print(config["learning_rate"])
```

---

## Creating Datasets from Scratch

For learning ML, you'll often create **synthetic datasets**:

```python
import numpy as np

# === LINEAR DATA ===
# y = 2x + 1 + noise
np.random.seed(42)
n_samples = 200
X = np.random.uniform(-5, 5, n_samples)
y = 2 * X + 1 + np.random.normal(0, 0.5, n_samples)

print(f"Features shape: {X.shape}")   # (200,)
print(f"Targets shape: {y.shape}")    # (200,)

# === CLASSIFICATION DATA ===
# Two classes: class 0 centered at (-2, -2), class 1 at (2, 2)
n_per_class = 100

class_0 = np.random.randn(n_per_class, 2) + np.array([-2, -2])
class_1 = np.random.randn(n_per_class, 2) + np.array([2, 2])

X_clf = np.vstack([class_0, class_1])          # (200, 2)
y_clf = np.array([0]*n_per_class + [1]*n_per_class)  # (200,)

print(f"Classification X: {X_clf.shape}")  # (200, 2)
print(f"Classification y: {y_clf.shape}")  # (200,)
```

---

## Train/Test Split

**Never** evaluate your model on the same data you trained on:

```python
import numpy as np

def train_test_split(X, y, test_ratio=0.2, seed=42):
    """Split data into training and testing sets."""
    np.random.seed(seed)
    n = len(X)
    indices = np.random.permutation(n)
    
    test_size = int(n * test_ratio)
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]
    
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

# Usage
X_train, X_test, y_train, y_test = train_test_split(X, y, test_ratio=0.2)

print(f"Training: {len(X_train)} samples")  # 160
print(f"Testing:  {len(X_test)} samples")   # 40
```

!!! warning "Golden Rule"
    Always split BEFORE training. If you train on test data, your results are meaningless.

---

## Data Normalization

ML models work better when data is **scaled** to a standard range:

```python
import numpy as np

# === MIN-MAX NORMALIZATION ===
# Scales data to [0, 1]
def normalize(X):
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    return (X - X_min) / (X_max - X_min + 1e-8)

# === STANDARDIZATION ===
# Scales data to mean=0, std=1
def standardize(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    return (X - mean) / (std + 1e-8)

# Example
data = np.array([100, 200, 300, 400, 500], dtype=float)
print(f"Original:     {data}")
print(f"Normalized:   {normalize(data)}")       # [0. 0.25 0.5 0.75 1.]
print(f"Standardized: {standardize(data)}")     # [-1.41 -0.71 0. 0.71 1.41]
```

!!! tip "When to normalize?"
    - **Always** when features have very different scales (e.g., age 0-100 vs salary 0-1,000,000)
    - **Usually** for neural networks — they converge faster with normalized data
    - **Sometimes** for tree-based methods — they don't need it but it doesn't hurt

---

## Shuffling Data

```python
import numpy as np

X = np.array([1, 2, 3, 4, 5])
y = np.array([10, 20, 30, 40, 50])

# Shuffle both arrays the same way
indices = np.random.permutation(len(X))
X_shuffled = X[indices]
y_shuffled = y[indices]

print(f"X: {X_shuffled}")  # e.g., [3 1 5 2 4]
print(f"y: {y_shuffled}")  # e.g., [30 10 50 20 40]
```

---

## Mini-Batches

For large datasets, we train on small chunks called **mini-batches**:

```python
import numpy as np

def get_batches(X, y, batch_size=32):
    """Yield mini-batches of data."""
    n = len(X)
    indices = np.random.permutation(n)
    
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_idx = indices[start:end]
        yield X[batch_idx], y[batch_idx]

# Usage
X = np.random.randn(100)
y = np.random.randn(100)

for batch_X, batch_y in get_batches(X, y, batch_size=32):
    print(f"Batch size: {len(batch_X)}")
# Output: 32, 32, 32, 4
```

---

## One-Hot Encoding

For classification with multiple classes:

```python
import numpy as np

def one_hot_encode(labels, num_classes=None):
    """Convert integer labels to one-hot vectors."""
    if num_classes is None:
        num_classes = labels.max() + 1
    
    one_hot = np.zeros((len(labels), num_classes))
    one_hot[np.arange(len(labels)), labels] = 1
    return one_hot

# Example: 3 classes (0, 1, 2)
labels = np.array([0, 1, 2, 1, 0])
encoded = one_hot_encode(labels)
print(encoded)
# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]
#  [0. 1. 0.]
#  [1. 0. 0.]]
```

---

## Putting It All Together

Here's a complete data preparation pipeline:

```python
import numpy as np

# 1. Generate data
np.random.seed(42)
n = 500
X = np.random.randn(n, 3)  # 500 samples, 3 features
true_weights = np.array([2.0, -1.0, 0.5])
y = X @ true_weights + 0.1 * np.random.randn(n)  # Linear relationship + noise

# 2. Split
train_size = int(0.8 * n)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 3. Normalize
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train_norm = (X_train - mean) / std
X_test_norm = (X_test - mean) / std  # Use TRAIN statistics!

# 4. Summary
print(f"Training set:  {X_train_norm.shape}")    # (400, 3)
print(f"Testing set:   {X_test_norm.shape}")     # (100, 3)
print(f"Train mean: {X_train_norm.mean(axis=0)}")  # ≈ [0, 0, 0]
print(f"Train std:  {X_train_norm.std(axis=0)}")   # ≈ [1, 1, 1]
```

!!! warning "Important"
    Always fit normalization on **training data only**, then apply the same transformation to test data. Otherwise you're "leaking" test information.

---

**Next:** [What is Machine Learning? →](../ml-fundamentals/what-is-ml.md)
