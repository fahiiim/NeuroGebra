# API Reference

Complete reference for every class and function in Neurogebra.

!!! tip "New here?"
    Start with the [tutorials](../tutorial/mathforge.md) first, then come back here when you need details.

---

## Quick Import Guide

```python
# Core imports — you'll use these the most
from neurogebra import MathForge, Expression

# Autograd (manual neural networks)
from neurogebra.core.autograd import Value, Tensor

# Training
from neurogebra.core.trainer import Trainer

# Model building
from neurogebra.builders.model_builder import ModelBuilder

# Educational interface
from neurogebra.core.neurocraft import NeuroCraft

# Datasets
from neurogebra.datasets.loaders import generate_linear, generate_spirals
```

---

## Core Classes

### Expression

The fundamental building block — a mathematical expression with symbolic and numerical capabilities.

::: neurogebra.core.expression.Expression
    options:
      show_root_heading: true
      show_source: true

---

### MathForge

Your gateway to 50+ pre-built mathematical expressions organized by category.

::: neurogebra.core.forge.MathForge
    options:
      show_root_heading: true
      show_source: true

---

### Trainer

Fits trainable expressions to data using SGD or Adam optimization.

::: neurogebra.core.trainer.Trainer
    options:
      show_root_heading: true
      show_source: true

---

### Value (Autograd)

Scalar value with automatic differentiation — the engine behind backpropagation.

::: neurogebra.core.autograd.Value
    options:
      show_root_heading: true
      show_source: true

---

### Tensor (Autograd)

Multi-dimensional array with gradient tracking for batch operations.

::: neurogebra.core.autograd.Tensor
    options:
      show_root_heading: true
      show_source: true

---

## Builders

### ModelBuilder

Keras-like interface for building neural network architectures layer by layer.

::: neurogebra.builders.model_builder.ModelBuilder
    options:
      show_root_heading: true
      show_source: true

---

## Repository

Pre-built expression collections organized by category.

### Activations

::: neurogebra.repository.activations.get_activations
    options:
      show_root_heading: true

### Losses

::: neurogebra.repository.losses.get_losses
    options:
      show_root_heading: true

### Regularizers

::: neurogebra.repository.regularizers.get_regularizers
    options:
      show_root_heading: true

### Algebra

::: neurogebra.repository.algebra.get_algebra_expressions
    options:
      show_root_heading: true

### Calculus

::: neurogebra.repository.calculus.get_calculus_expressions
    options:
      show_root_heading: true

### Statistics

::: neurogebra.repository.statistics.get_statistics_expressions
    options:
      show_root_heading: true

### Linear Algebra

::: neurogebra.repository.linalg.get_linalg_expressions
    options:
      show_root_heading: true

### Metrics

::: neurogebra.repository.metrics.get_metrics_expressions
    options:
      show_root_heading: true

###Transforms

::: neurogebra.repository.transforms.get_transforms_expressions
    options:
      show_root_heading: true

### Optimization

::: neurogebra.repository.optimization.get_optimization_expressions
    options:
      show_root_heading: true

---

## Visualization

### Plotting

::: neurogebra.viz.plotting
    options:
      show_root_heading: true

### Interactive

::: neurogebra.viz.interactive
    options:
      show_root_heading: true

---

## Utilities

### Helpers

::: neurogebra.utils.helpers
    options:
      show_root_heading: true

### Explain

::: neurogebra.utils.explain.ExpressionExplainer
    options:
      show_root_heading: true

---

## Framework Bridges

Convert Neurogebra expressions to production frameworks.

### PyTorch Bridge

::: neurogebra.bridges.pytorch_bridge
    options:
      show_root_heading: true

### TensorFlow Bridge

::: neurogebra.bridges.tensorflow_bridge
    options:
      show_root_heading: true

### JAX Bridge

::: neurogebra.bridges.jax_bridge
    options:
      show_root_heading: true
