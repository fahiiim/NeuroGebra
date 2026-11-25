# Release Notes & Changelog

**Current Release:** **v2.8.3** (2026-11-23)

---

## Release Highlights (v2.8.0 &ndash; v2.8.3)

### v2.8.3 &mdash; Cross-Platform Python 3.9 Compatibility & Docs Overhaul
- Restored full Python 3.9 runtime compatibility across Linux (`ubuntu-22.04`), Windows, and macOS CI runners by converting PEP 604 runtime union type hints (`|`) to PEP 563 (`from __future__ import annotations`) and `typing.Optional`.
- Organized documentation with a PyTorch-style hierarchy, NeuroGebra logo teal (`#008484`), black (`#0A0B0B`), and white (`#FFFFFF`) design system, full API coverage, and up-to-date tutorials for all v2.8 subsystems.

### v2.8.2 &mdash; Rich LaTeX Rendering & Mathematical Audit Hardening (V01&ndash;V20)
- **Rich LaTeX Math Rendering**: Added `_repr_latex_()`, `latex()`, and `to_latex()` on `Expression` plus rich `display(Math(...))` outputs in [Project 13 Notebook](https://github.com/fahiiim/NeuroGebra/blob/main/examples/projects/13-symbolic-autograd-and-higher-order-derivatives.ipynb).
- **Optimizer Integration (V01)**: `Model.compile` accepts standalone `Optimizer` instances (`SGD`, `Adam`, `RMSprop`) and string `"rmsprop"`, stepped via `LayerParam` wrappers.
- **Scheduler Stepping (V02)**: Guaranteed epoch-end stepping in both `Model.fit` and `EducationalTrainer.train`.
- **Exact LayerNorm VJP (V03)**: Feature-scaled analytical backward pass verified against numerical finite differences.
- **PINN Multi-Parameter Gradients (V04)**: Differential equation residual loss differentiates both weights and biases.
- **Complete Serialization & Optimizer Moments (V05)**: `Model.save()` and `Model.load()` persist `ResidualDense` weights/skips, `LayerNorm` parameters, running statistics, step counters, and initialize moment buffers cleanly.
- **Numerical & Autograd Hardening (V06&ndash;V15)**: Macro F1 per-class averaging, leaf gradient accumulation preservation, negative-axis `Tensor.transpose` normalization, Softmax underflow stability, sample-weighted batch metrics, asymmetric even-kernel `Conv2D` same-padding, modern bridge layers (`LayerNorm`, `GlobalAveragePooling2D`, `ResidualDense`), and Softplus / Box-Cox stability.

### v2.8.1 &mdash; Enterprise Emoji Elimination & Hardened CI Matrix
- Eliminated all emoji characters across code, docstrings, markdown documentation, examples, and configurations.
- Modernized CI workflows across Ubuntu, Windows, and Apple Silicon macOS runners for Python 3.9&ndash;3.12.

### v2.8.0 &mdash; SciML, Schedulers, Zero-Dep Metrics, Modern Layers & 20 GPU Notebooks
- Introduced `neurogebra.science.PINN` for differential equation solving.
- Introduced `neurogebra.optim` standalone optimizers (`SGD`, `Adam`, `RMSprop`) and LR schedulers (`StepLR`, `CosineAnnealingLR`, `ExponentialLR`, `ReduceLROnPlateau`).
- Introduced `neurogebra.metrics` (12 classification and regression evaluation functions with zero extra dependencies).
- Added `LayerNorm`, `GlobalAveragePooling2D`, `ResidualDense`, and `Model.add()` chaining to `ModelBuilder`.
- Authored and executed **20 GPU notebook projects** (`examples/projects/01`&ndash;`20`).

### v2.6.0 &ndash; v2.7.0 &mdash; Native C-JIT & CUDA GPU Engine + 48 Datasets
- Added `neurogebra.acceleration` compiling symbolic expressions dynamically into AVX2 SIMD C++ shared objects (up to 2.04x speedup) and NVIDIA CUDA kernels via NVRTC (up to 4.76x speedup on RTX 5070).
- Expanded built-in dataset catalog to 48 loaders across `Datasets`, `ExpandedDatasets`, and `CombinedDatasets`.
- Added `Model.load()`, `Model.train_on_batch()`, full `BatchNorm` VJP, and public `neurogebra.bridges` exports (`to_pytorch`, `model_to_pytorch`, `to_tensorflow`, `model_to_tensorflow`, `to_jax`).

---

## Install Latest Release

```bash
pip install --upgrade neurogebra
```

- **PyPI Project**: [https://pypi.org/project/neurogebra/2.8.3/](https://pypi.org/project/neurogebra/2.8.3/)
- **Full Changelog**: [CHANGELOG.md on GitHub](https://github.com/fahiiim/NeuroGebra/blob/main/CHANGELOG.md)
