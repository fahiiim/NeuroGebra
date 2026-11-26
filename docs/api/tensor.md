---
level: API reference
runtime: Python / NumPy
---

# Tensor

NumPy-backed tensor operations and eager gradient graphs.

Read the [autograd guide](../tutorial/autograd.md). Gradients accumulate at leaves; rebuild the graph after updating parameters and avoid mutating values between forward and backward.

::: neurogebra.core.autograd.Tensor
    options:
      show_root_heading: true
      show_source: false
