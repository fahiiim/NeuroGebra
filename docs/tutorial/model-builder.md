# ModelBuilder

`ModelBuilder` provides an **educational, intuitive interface** for building neural network architectures — similar to Keras, but with built-in explanations.

---

## What is ModelBuilder?

ModelBuilder lets you:

- Build neural networks layer by layer
- Understand what each layer does
- Get explanations and best-practice suggestions
- Use pre-built templates for common tasks

---

## Quick Start

```python
from neurogebra import ModelBuilder

builder = ModelBuilder()

# Build a simple classifier
model = builder.Sequential([
    builder.Dense(128, activation="relu"),
    builder.Dropout(0.2),
    builder.Dense(64, activation="relu"),
    builder.Dense(10, activation="softmax")
])

# See what you built
model.summary()
```

---

## Layer Types

### Dense (Fully Connected)

Every neuron connects to every neuron in the next layer.

```python
# Basic dense layer
layer = builder.Dense(128, activation="relu")

# With input shape (first layer only)
first_layer = builder.Dense(128, activation="relu", input_shape=(784,))

# Learn what it does
layer.explain()
```

Output of `explain()`:
```
 DENSE Layer
   Fully connected layer - every neuron connects to every neuron in next layer

   Best used for:
   • Classification
   • Regression
   • Final output layer

   Neurons/Units: 128
   Activation: relu
```

### Dropout

Randomly deactivates neurons during training to prevent overfitting.

```python
dropout = builder.Dropout(rate=0.2)  # Drop 20% of neurons
dropout.explain()
```

### Conv2D (Convolutional)

Detects patterns in images using sliding filters.

```python
conv = builder.Conv2D(filters=32, kernel_size=3, activation="relu")
conv.explain()
```

### MaxPooling2D

Reduces image dimensions by taking maximum values in regions.

```python
pool = builder.MaxPooling2D(pool_size=2)
pool.explain()
```

### Flatten

Converts multi-dimensional data to 1D for dense layers.

```python
flatten = builder.Flatten()
```

### BatchNorm

Normalizes layer outputs to stabilize training.

```python
bn = builder.BatchNorm()
bn.explain()
```

---

## Building Models

### Sequential Model

Layers stacked one after another:

```python
# Simple classifier
classifier = builder.Sequential([
    builder.Dense(128, activation="relu"),
    builder.Dropout(0.2),
    builder.Dense(64, activation="relu"),
    builder.Dense(10, activation="softmax")
], name="my_classifier")

classifier.summary()
```

### From Template

Use pre-built architectures:

```python
# See available templates
builder.list_templates()

# Create from template
model = builder.from_template("simple_classifier")
model.summary()

# Other templates:
# - "simple_classifier"  → Basic dense network
# - "image_classifier"   → CNN for images
# - "regression"         → For predicting numbers
# - "binary_classifier"  → For yes/no tasks
```

---

## Model Templates

### Simple Classifier

```python
model = builder.from_template("simple_classifier")
model.explain_architecture()
```

Architecture:
```
Dense(128, relu) → Dropout(0.2) → Dense(64, relu) → Dense(10, softmax)
```

### Image Classifier (CNN)

```python
model = builder.from_template("image_classifier")
model.explain_architecture()
```

Architecture:
```
Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Flatten → Dense(128) → Dense(10)
```

### Regression Network

```python
model = builder.from_template("regression")
```

Architecture:
```
Dense(64, relu) → Dense(32, relu) → Dense(1, linear)
```

### Binary Classifier

```python
model = builder.from_template("binary_classifier")
```

Architecture:
```
Dense(64, relu) → Dropout(0.3) → Dense(32, relu) → Dense(1, sigmoid)
```

---

## Architecture Suggestions

ModelBuilder can suggest architectures based on your task:

```python
# Get suggestion for a task
builder.suggest_architecture(
    task="classification",
    input_size=784,
    output_size=10
)
```

---

## Understanding Your Model

### Summary

```python
model.summary()
```

Shows the layer structure with types, units, and activations.

### Explain Architecture

```python
model.explain_architecture()
```

Gives plain-English explanation of what each layer does and why it's there.

---

## Building an Image Classifier Step by Step

```python
from neurogebra import ModelBuilder

builder = ModelBuilder()

# Step 1: Feature extraction (convolutions)
# Step 2: Dimensionality reduction (pooling)
# Step 3: Classification (dense layers)

model = builder.Sequential([
    # Feature extraction
    builder.Conv2D(32, kernel_size=3, activation="relu"),
    builder.MaxPooling2D(pool_size=2),
    builder.Conv2D(64, kernel_size=3, activation="relu"),
    builder.MaxPooling2D(pool_size=2),
    
    # Bridge
    builder.Flatten(),
    
    # Classification
    builder.Dense(128, activation="relu"),
    builder.Dropout(0.5),
    builder.Dense(10, activation="softmax")
], name="mnist_classifier")

# Understand it
model.summary()
model.explain_architecture()
```

---

## Layer Selection Guide

| Task | Layers to Use |
|------|--------------|
| Tabular data classification | Dense → Dropout → Dense → Softmax |
| Image classification | Conv2D → MaxPool → Dense → Softmax |
| Binary classification | Dense → Dense → Dense(1) → Sigmoid |
| Regression | Dense → Dense → Dense(1) → Linear |

| Layer | When to Use |
|-------|-------------|
| Dense | Always (at least for the output) |
| Dropout | When you have overfitting |
| Conv2D | For images/spatial data |
| MaxPooling2D | After Conv2D to reduce size |
| Flatten | Between Conv and Dense |
| BatchNorm | For deeper networks to stabilize training |

---

**Next:** [NeuroCraft Interface →](neurocraft.md)
