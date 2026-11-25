# API Reference

Welcome to the NeuroGebra API Reference. NeuroGebra is structured into modular subsystems designed for symbolic mathematics, hardware acceleration, scientific machine learning, and deep learning observability.

---

## Navigation & Subsystems

<div class="ng-grid">
  <div class="ng-card">
    <div class="ng-card-title">
      <a href="core.md">neurogebra.core</a>
    </div>
    <div class="ng-card-desc">
      Foundational mathematical engine: <code>Expression</code>, <code>MathForge</code> (285 formulas), micro-autograd (<code>Value</code>, <code>Tensor</code>), <code>Trainer</code>, and interactive <code>NeuroCraft</code>.
    </div>
  </div>

  <div class="ng-card">
    <div class="ng-card-title">
      <a href="builders.md">neurogebra.builders</a>
    </div>
    <div class="ng-card-desc">
      Neural network construction with <code>ModelBuilder</code>, <code>Model</code>, classic layers, and modern layers (<code>LayerNorm</code>, <code>GlobalAveragePooling2D</code>, <code>ResidualDense</code>).
    </div>
  </div>

  <div class="ng-card">
    <div class="ng-card-title">
      <a href="acceleration.md">neurogebra.acceleration</a>
    </div>
    <div class="ng-card-desc">
      Hardware acceleration subsystem: C-JIT native compiler with AVX2 SIMD loop vectorization and CUDA GPU kernels via NVRTC.
    </div>
  </div>

  <div class="ng-card">
    <div class="ng-card-title">
      <a href="science.md">neurogebra.science</a>
    </div>
    <div class="ng-card-desc">
      Scientific computing and Physics-Informed Neural Networks (<code>PINN</code>) with automatic first/second-order spatial & temporal derivatives.
    </div>
  </div>

  <div class="ng-card">
    <div class="ng-card-title">
      <a href="optim.md">neurogebra.optim</a>
    </div>
    <div class="ng-card-desc">
      Standalone optimizers (<code>SGD</code> with Nesterov, <code>Adam</code>, <code>RMSprop</code>) and dynamic LR schedulers (<code>StepLR</code>, <code>CosineAnnealingLR</code>, <code>ExponentialLR</code>, <code>ReduceLROnPlateau</code>).
    </div>
  </div>

  <div class="ng-card">
    <div class="ng-card-title">
      <a href="metrics.md">neurogebra.metrics</a>
    </div>
    <div class="ng-card-desc">
      Zero-dependency evaluation metrics: accuracy, precision, recall, F1, confusion matrix, classification report, MSE, RMSE, MAE, R², explained variance.
    </div>
  </div>

  <div class="ng-card">
    <div class="ng-card-title">
      <a href="logging.md">neurogebra.logging & training</a>
    </div>
    <div class="ng-card-desc">
      Observatory Pro real-time telemetry, <code>AdaptiveLogger</code> anomaly detection, <code>TrainingFingerprint</code> reproducibility, and <code>EducationalTrainer</code>.
    </div>
  </div>

  <div class="ng-card">
    <div class="ng-card-title">
      <a href="bridges-datasets.md">bridges & datasets</a>
    </div>
    <div class="ng-card-desc">
      Zero-overhead framework bridges to PyTorch, TensorFlow, and JAX, plus 48 built-in dataset loaders for immediate experimentation.
    </div>
  </div>

  <div class="ng-card">
    <div class="ng-card-title">
      <a href="repository.md">repository & viz</a>
    </div>
    <div class="ng-card-desc">
      Pre-built expression registries across 10 mathematical domains and static/interactive 2D/3D visualization utilities.
    </div>
  </div>
</div>

---

## Quick Import Guide

```python
# Core mathematical foundation
from neurogebra import MathForge, Expression

# Micro-autograd engine
from neurogebra.core.autograd import Value, Tensor

# Neural network construction & modern layers
from neurogebra.builders.model_builder import ModelBuilder, Model

# Hardware acceleration
from neurogebra.acceleration import compile_c, compile_cuda, get_acceleration_info

# Scientific ML / Physics-Informed Neural Networks
from neurogebra.science import PINN

# Optimizers & learning rate schedulers
from neurogebra.optim import Adam, SGD, CosineAnnealingLR, ReduceLROnPlateau

# Zero-dependency evaluation metrics
from neurogebra.metrics import accuracy_score, f1_score, classification_report, r2_score

# Training Observatory Pro & telemetry
from neurogebra.logging import AdaptiveLogger, TrainingFingerprint, DashboardExporter

# Framework bridges
from neurogebra.bridges import to_pytorch, to_tensorflow, to_jax

# Built-in datasets (48 loaders)
from neurogebra.datasets import Datasets, ExpandedDatasets
```
