# Optimizers & Learning Rate Schedulers

The `neurogebra.optim` module provides first-class gradient-based optimizers and learning rate schedulers compatible with both `ModelBuilder` sequential models and custom training loops.

---

## Optimizers

### SGD (Stochastic Gradient Descent)

::: neurogebra.optim.optimizers.SGD
    options:
      show_root_heading: true
      show_source: false

### Adam (Adaptive Moment Estimation)

::: neurogebra.optim.optimizers.Adam
    options:
      show_root_heading: true
      show_source: false

### RMSprop

::: neurogebra.optim.optimizers.RMSprop
    options:
      show_root_heading: true
      show_source: false

---

## Learning Rate Schedulers

### StepLR

::: neurogebra.optim.lr_scheduler.StepLR
    options:
      show_root_heading: true
      show_source: false

### CosineAnnealingLR

::: neurogebra.optim.lr_scheduler.CosineAnnealingLR
    options:
      show_root_heading: true
      show_source: false

### ExponentialLR

::: neurogebra.optim.lr_scheduler.ExponentialLR
    options:
      show_root_heading: true
      show_source: false

### ReduceLROnPlateau

::: neurogebra.optim.lr_scheduler.ReduceLROnPlateau
    options:
      show_root_heading: true
      show_source: false
