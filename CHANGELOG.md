# Changelog

All notable changes to Neurogebra will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.1] - 2026-02-16

### Added — Training Observatory 🔭

The headline feature of v1.2.1 is the **Training Observatory**: an advanced training logging and visualization system that shows the complete mathematical picture of what happens inside your neural network during training — in real time, in colour, in your terminal.

#### New Module: `neurogebra.logging`
- **`TrainingLogger`** — Event-driven, multi-level logger with pluggable backends
- **`LogConfig`** — Preset-based configuration (minimal, standard, verbose, research, production)
- **`TerminalDisplay`** — Rich-based colour-coded terminal renderer with severity icons, sparkline charts, gradient heatmaps, and weight histograms
- **`FormulaRenderer`** — Unicode and LaTeX rendering of forward/backward pass equations and loss function formulas
- **`GradientMonitor`** — Real-time gradient norm tracking with vanishing/exploding detection
- **`WeightMonitor`** — Weight distribution histograms, dead neuron detection, change tracking
- **`ActivationMonitor`** — Dead ReLU detection, sigmoid/tanh saturation analysis
- **`PerformanceMonitor`** — Per-layer and per-epoch timing, bottleneck detection
- **`SmartHealthChecker`** — Automatic detection of 8+ training problems (NaN, overfitting, underfitting, stagnation, divergence, vanishing/exploding gradients, dead neurons, activation saturation) with actionable recommendations
- **`ComputationGraph` (GraphTracker)** — Full DAG tracking of every operation with shapes, values, gradients, and formulas
- **`ImageLogger`** — ASCII art rendering of input images, activation maps, filter weights, and saliency maps in terminal
- **4 Export Backends** — JSONExporter, CSVExporter, HTMLExporter (with interactive Chart.js graphs), MarkdownExporter

#### Enhanced Model Builder
- **Real Forward/Backward Passes** — Layer class now performs actual matrix multiplications, activation functions, and gradient computation using NumPy
- **He Weight Initialization** — Proper weight initialization for each layer type
- **Adam & SGD Optimizers** — Built-in optimizer implementations at the layer level
- **8 Activation Functions** — relu, sigmoid, tanh, softmax, leaky_relu, elu, swish, gelu — all with correct forward and backward implementations
- **Observatory Integration** — `model.compile(log_level="expert")` enables full logging with one argument
- **Real Loss Computation** — MSE, MAE, binary cross-entropy, categorical cross-entropy with analytical gradients
- **Model Save** — Pickle-based model serialization including weights

#### New Dependencies
- `rich>=13.0.0` — Rich terminal display
- `colorama>=0.4.6` — Cross-platform ANSI colour support
- Optional: `tensorboard>=2.12.0`, `wandb>=0.15.0` (via `pip install neurogebra[logging]`)

### Changed
- Version bumped from 0.3.0 to 1.2.1
- `Model.fit()` now performs real training when Observatory is active (falls back to EducationalTrainer otherwise)
- `Model.predict()` uses real forward pass when layers are initialised
- `Model.evaluate()` computes actual loss and accuracy metrics
- Updated `__init__.py` to export `TrainingLogger`, `LogLevel`, `LogEvent`, `LogConfig`
- Updated `pyproject.toml` with new dependencies and `[logging]` optional extra

### Tests
- Added `tests/test_logging/test_observatory.py` — 35 comprehensive tests covering logger, config, monitors, health checks, exporters, computation graph, and full model integration

[1.2.1]: https://github.com/fahiiim/NeuroGebra/releases/tag/v1.2.1

## [0.3.0] - 2026-02-10

### Added — Complete Documentation Overhaul

- **30+ new documentation pages** — W3Schools-style progressive tutorial from absolute basics to advanced projects
- **Getting Started section** (3 pages): Installation, First Program, How Neurogebra Works
- **Python for ML Refresher** (3 pages): Python Basics, NumPy Essentials, Data Handling
- **ML Fundamentals** (4 pages): What is ML, Types of ML, ML Workflow, Math Behind ML
- **Neurogebra Tutorial** (12 pages): MathForge, Expressions, Activations, Losses, Gradients, Composition, Training, Autograd, Tensors, ModelBuilder, NeuroCraft, Datasets
- **Advanced Topics** (6 pages): Custom Expressions, Framework Bridges, Visualization, Regularization, Optimization, Performance Tips
- **3 Full Projects with PyTorch side-by-side comparison**:
  - Project 1: Linear Regression — Neurogebra vs PyTorch
  - Project 2: Image Classifier — Neurogebra vs PyTorch
  - Project 3: Neural Network from Scratch — Neurogebra vs PyTorch
- **Expanded API Reference** with all repository modules (calculus, statistics, linalg, metrics, transforms, optimization)

### Changed

- Completely rewritten `mkdocs.yml` with new 8-section navigation, Material theme with deep purple/amber colors, tabbed content support, MathJax for math equations
- Rewritten landing page (`index.md`) with comparison table, quick example, and section overview
- Updated version from 0.2.2 to 0.3.0

[0.3.0]: https://github.com/fahiiim/NeuroGebra/releases/tag/v0.3.0

## [0.2.2] - 2026-02-09

### Fixed

- Fixed missing typing imports (Dict, List, Any) in expanded_loaders.py that caused NameError on import
- Critical bugfix for ExpandedDatasets functionality

[0.2.2]: https://github.com/fahiiim/NeuroGebra/releases/tag/v0.2.2

## [0.2.1] - 2026-02-09

### Fixed

- Fixed logo display on PyPI by using absolute GitHub URL instead of relative path
- Logo now properly displays on PyPI project page

[0.2.1]: https://github.com/fahiiim/NeuroGebra/releases/tag/v0.2.1

## [0.2.0] - 2026-02-09

### Added

- **100+ Dataset Collection** with 38+ working datasets across 4 categories
- **ExpandedDatasets class** with 27 additional dataset loaders:
  - Classification: Covtype, Letter Recognition, Optical Recognition, Pendigits, Satimage, Segment, Shuttle, Vehicle, Vowel, Connect-4
  - Regression: Energy Efficiency, Power Plant, Yacht Hydrodynamics, Abalone, Airfoil Self-Noise, Wine Quality Red
  - Synthetic Patterns: Spirals, Checkerboard, Blobs, Swiss Roll, S-Curve, Half Kernel
  - Time Series: Sine Wave, Random Walk, Stock Prices, Seasonal Data, AR Process
- **CombinedDatasets class** unified interface for all datasets
- **Dataset discovery tools:**
  - `Datasets.list_all()` - Browse all available datasets with formatted output
  - `Datasets.search(keyword)` - Search datasets by topic/category
  - `Datasets.get_info(name)` - Detailed dataset information
- **Educational metadata** for every dataset (difficulty, use cases, descriptions)
- **Verbose mode** for dataset loaders with sample counts and descriptions
- **New documentation files:**
  - `PUBLISHING.md` - Complete guide for publishing to PyPI
  - `RELEASE_NOTES.md` - Detailed release notes
  - `DATASETS_STATUS.md` - Implementation tracking
  - `publish.ps1` - PowerShell publishing script
- **Enhanced MANIFEST.in** to include logo and new documentation
- **scikit-learn integration** as optional dependency for real-world datasets

### Changed

- Updated package version from 0.1.1 to 0.2.0
- Enhanced `loaders.py` with utility methods
- Updated README.md with Datasets section and Neurogebra logo
- Updated `datasets/__init__.py` with new exports

### Infrastructure

- Added `datasets` optional dependency group with scikit-learn
- Updated `all` optional dependencies to include datasets
- Added logo assets folder

[0.2.0]: https://github.com/fahiiim/NeuroGebra/releases/tag/v0.2.0

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
