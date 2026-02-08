# Changelog

All notable changes to Neurogebra will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-02-09

### Added

- **Core Expression class** with symbolic, numerical, and trainable support
- **MathForge** central hub for accessing 285 mathematical expressions
- **NeuroCraft** educational interface with tutorials and guided learning
- **ModelBuilder** beginner-friendly model construction with architecture templates
- **EducationalTrainer** training with real-time tips, warnings, and debugging advice
- **TutorialSystem** with 4 interactive step-by-step tutorials
- **Micro autograd engine** (Value and Tensor classes)
- **Trainer** with SGD and Adam optimizers
- **Datasets** module with MNIST, Iris, regression, XOR, moons, circles loaders
- **Repository of 285 mathematical expressions across 10 modules:**
  - Activations (15): ReLU, Sigmoid, Tanh, Swish, GELU, Mish, and more
  - Losses (8): MSE, MAE, Binary Cross-Entropy, Huber, Hinge, and more
  - Regularizers (20): L1, L2, Elastic Net, SCAD, MCP, Group Lasso, and more
  - Algebra (48): Polynomials, kernels, distributions, special functions
  - Calculus (48): Elementary functions, trig, hyperbolic, Taylor series, transforms
  - Statistics (35): PDFs, CDFs, information theory, Bayesian, regression
  - Linear Algebra (24): Norms, distances, projections, matrix ops, attention
  - Optimization (27): SGD, Adam, AdamW, LR schedules, loss landscapes
  - Metrics (27): Classification, regression, similarity, ranking, model selection
  - Transforms (33): Normalization, encoding, initialization, signal processing
- **Search engine** for finding expressions by name, description, or category
- **Explanation engine** with multiple output formats
- **Visualization** tools (matplotlib and plotly)
- **Framework bridges** for PyTorch, TensorFlow, JAX
- **Comprehensive test suite** with 377 passing tests
- **Full documentation** with MkDocs Material theme
- **CI/CD** with GitHub Actions (tests, docs, PyPI publishing)

### Infrastructure

- Project uses `hatchling` build backend
- Python 3.9+ support
- MIT License
- Comprehensive type hints and docstrings

[0.1.0]: https://github.com/fahiiim/NeuroGebra/releases/tag/v0.1.0
