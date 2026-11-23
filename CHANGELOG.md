# Changelog

All notable changes to Neurogebra will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.8.3] - 2026-11-23

### Fixed
- **Python 3.9 Compatibility Hardening**: Restored complete Python 3.9 runtime compatibility across Linux, Windows, and macOS CI runners by replacing PEP 604 runtime union syntax (`|`) with `from __future__ import annotations` and standard `typing.Optional` / `typing.Dict` / `typing.List` in `educational_trainer.py`, `pytorch_bridge.py`, `tensorflow_bridge.py`, and `model_builder.py`.

## [2.8.2] - 2026-11-23

### Added
- **Rich LaTeX Mathematical Expression Rendering**: Added `_repr_latex_()`, `latex()`, and `to_latex()` methods to `neurogebra.core.expression.Expression` with full support for Jupyter notebook rich display outputs (`display(Math(...))` and `text/latex` MIME representation).
- **Execution & LaTeX Display in Project 13**: Updated `examples/projects/13-symbolic-autograd-and-higher-order-derivatives.ipynb` with rich LaTeX outputs for objective functions, analytical gradient vectors, and Hessian matrices.

### Fixed & Hardened (Expert Mathematical Audit Findings V01-V20)
- **V01 (Optimizer Stepping)**: Model optimizer stepping now supports standalone optimizer objects (`RMSprop`, `Adam`, `SGD`) via `LayerParam` references (.data, .grad) and added `"rmsprop"` string support.
- **V02 (LR Scheduler Stepping)**: Learning rate schedulers are stepped reliably at the end of each epoch across all training paths including `EducationalTrainer`.
- **V03 (LayerNorm Analytical Backward VJP)**: Exact analytical vector-jacobian product for LayerNorm with feature-scaled upstream gradient $g = \text{grad\_output} \times \gamma$ inside normalization sums, validated via finite differences.
- **V04 (PINN Multi-Parameter Residual Gradients)**: Physics-informed loss backpropagation differentiates both weights and biases across all layers.
- **V05 (Serialization & Moment Persistence)**: Complete state dictionary serialization in `Model.save()` and `Model.load()` covering `ResidualDense`, `LayerNorm`, running statistics, and optimizer moments.
- **V06 (Macro F1 Metric)**: Corrected macro F1 metric calculation to compute per-class F1 before unweighted mean averaging.
- **V07 (Autograd Graph Accumulation)**: Non-leaf graph nodes reset during autograd traversal while leaf tensor gradients accumulate properly across backward calls.
- **V08 (Tensor Transpose Negative Axes)**: Negative axes normalized before inverse permutation calculation.
- **V09 (Softmax Gradient Underflow Stability)**: Added underflow mask safeguarding against extreme negative logits and infinite cross-entropy gradients.
- **V10 (Adam Step Counter Persistence)**: Step counter tracked across `train_on_batch()` and `fit()` cycles.
- **V11 (Sample Weighted Aggregation)**: Batch metrics weighted by actual batch sample sizes during epoch loss/metric aggregation.
- **V12 (Metric Input Validation)**: Automatic 1D/2D shape alignment and length validation across all classification and regression metrics.
- **V13 (Conv2D Asymmetric Padding)**: Asymmetric same-padding for even convolutional kernels ($k-1$ total pad).
- **V14 (Bridge Validation)**: Deep learning framework bridges support modern layers (`LayerNorm`, `GlobalAveragePooling2D`, `ResidualDense`) and fail closed with `NotImplementedError` for unsupported structures.
- **V15 (Formula Numerical Stability)**: Softplus and Box-Cox numerical evaluations guarded against overflow and division by zero ($\lambda \to 0 \implies \ln(x)$).
- **V16-V20 (Provenances, AST Checks, Dashboard XSS & Security)**: Accurately documented dataset distributions, validated Python documentation snippets via AST, HTML-escaped dashboard metadata, and removed emojis across all repository files.

## [2.8.1] - 2026-11-22

### Changed & Cleaned
- **Complete Emoji Elimination**: Removed all emojis across the entire repository (documentation, benchmarks, examples, source code, and configuration files), maintaining a clean, professional, enterprise-grade codebase.
- **CI/CD Matrix Modernization**: Modernized GitHub Actions CI test suite across Linux (Ubuntu 22.04 & latest), Windows, and macOS (Apple Silicon runners) testing Python 3.9, 3.10, 3.11, and 3.12.
- **Release Automation Hardening**: Added `--skip-existing` protection to PyPI upload workflows to ensure idempotent and reliable publishing.
- **GitHub Workflow Actions Update**: Upgraded all GitHub Action dependencies to current stable releases.

## [2.8.0] - 2026-11-22

### Added
- **20 Executed GPU Notebook Projects**: Completely authored and executed 20 production-grade Jupyter notebooks in `examples/projects/` on NVIDIA GeForce RTX 5070 GPU, featuring live stdout training logs, stored cell execution counts, and embedded high-resolution Matplotlib figures.
- **Physics-Informed Neural Networks (`neurogebra.science.PINN`)**: Added deep scientific machine learning solver enforcing ordinary and partial differential equation residuals (Harmonic Oscillator, Burgers' equation, Heat equation) with collocation point sampling and joint data-physics optimization.
- **Dynamic Learning Rate Schedulers (`neurogebra.optim`)**: Added `LRScheduler` base class, `StepLR`, `CosineAnnealingLR`, `ExponentialLR`, and `ReduceLROnPlateau` with full integration into `ModelBuilder.fit()`.
- **First-Class Optimizers (`neurogebra.optim`)**: Standalone `SGD`, `Adam`, and `RMSprop` optimizer classes with momentum, weight decay, and velocity state buffers.
- **Evaluation Metrics (`neurogebra.metrics`)**: Zero-dependency pure Python/NumPy evaluation tools including `accuracy_score`, `confusion_matrix`, `precision_score`, `recall_score`, `f1_score`, `classification_report`, `r2_score`, `mean_squared_error`, `root_mean_squared_error`, `mean_absolute_error`, and `explained_variance_score`.
- **Modern Neural Network Layers**: Added `LayerNorm`, `GlobalAveragePooling2D`, and `ResidualDense` (skip-connection residual blocks) to `ModelBuilder`.
- **Model Construction Chaining**: Added `Model.add(layer)` method for intuitive layer stacking.

### Changed & Cleaned
- **Complete Emoji Removal**: Safely eliminated all emojis across the entire codebase (logs, docstrings, interactive CLI displays, print statements) and documentation (`README.md`), establishing a sleek, professional enterprise aesthetic.
- **Self-Contained Data Loaders in Projects**: Replaced fragile external dataset downloaders with NeuroGebra's robust built-in loaders (`load_fashion_mnist`, `load_mnist`, `load_california_housing`, `load_spam`, `load_digits`), ensuring 100% offline reproducibility.

## [2.7.0] - 2026-11-22

### Added
- **15 New Built-in Dataset Loaders**: Complete 48-dataset catalog covering Wine, Breast Cancer, Digits, Fashion-MNIST, Spam, Titanic, Credit Default, Mushroom, Adult Income, California Housing, Diabetes, Boston Housing, Auto MPG, Concrete Strength, and Bike Sharing with zero global seed pollution.
- **Model Checkpointing & Persistence**: Added `Model.load(filepath)` complementing `Model.save()`, supporting weights, layer configurations, and serialization integrity.
- **Model Batch API**: Added `Model.train_on_batch(X, y)` for step-by-step training.
- **Full BatchNorm Contract**: Full implementation including learned affine parameters ($\gamma, \beta$), running mean/variance tracking, momentum, $\epsilon$, explicit training/eval mode execution, and exact analytical backward pass.
- **Export of Framework Bridges**: Public exports of `to_pytorch`, `model_to_pytorch`, `to_tensorflow`, `model_to_tensorflow`, and `to_jax` from `neurogebra.bridges`.

### Changed & Fixed
- **Loss Reduction & Layer Gradients**: Removed redundant batch averaging in `Dense` and `Conv2D` backward passes, aligning layer gradients exactly with finite differences.
- **Softmax Vector-Jacobian Product**: Implemented exact analytical VJP $dz = a \odot (g - \sum(g \odot a))$ supporting arbitrary downstream losses and intermediate softmax layers.
- **GELU Derivative**: Analytical derivative approximation for GELU activation matching forward pass.
- **Autograd Engine**: Rank-promoted Tensor matrix multiplication (`@`) with NumPy-compatible gradient unbroadcasting across all dimensional permutations; zero-gradient reset on graph reuse.
- **Hardware Acceleration (C-JIT & CUDA-JIT)**: Added full NumPy broadcasting compatibility, empty array handling, C variable mangling (`__ng_v_k`, `__ng_idx`), baseline CPU compilation fallback, and strict GPU resource cleanup.
- **Observatory & Logging**: Early-exit guards on fine-grained metrics, production logging configuration preset (`LogLevel.BASIC` + silent console), XSS-safe HTML escaping, and UTF-8 file exporters.
- **CI & Release Engineering**: Aligned GitHub Actions to `@v4` / `@v5`, added `macos-13` runner support for Python 3.9, and updated security policy in `SECURITY.md`.

## [2.6.1] - 2026-11-21

### Fixed

- **Cross-Platform C-JIT Compilation**:
 - Resolved macOS Apple Silicon (ARM64) Clang flag error by gating `-mavx2` exclusively to x86_64 architectures.
 - Added `-dynamiclib` and `.dylib` native shared library handling on Darwin/macOS.
 - Implemented dynamic runtime toolchain verification in `is_c_available()` to prevent CI and test failures on environments with incomplete C compilers.
- **CI Matrix Compatibility**:
 - Aligned Ubuntu Python 3.9 runner to `ubuntu-22.04` to resolve Noble Numbat 24.04 toolcache deprecation in GitHub Actions.

## [2.6.0] - 2026-11-21

### Added

- **Hardware Acceleration Engine (`neurogebra.acceleration`)**:
 - `c_jit`: On-the-fly compilation of SymPy mathematical expressions into native C/C++ shared libraries with Clang/GCC, OpenMP, and AVX2 SIMD vectorization.
 - `cuda_jit`: Direct generation and NVRTC compilation of fused CUDA kernels running natively on NVIDIA GPUs (RTX 5070, 40-series, 30-series).
 - Multi-backend evaluation on `Expression`: `expr.compile("c")` and `expr.compile("cuda")`.
 - Up to 2.04x speedup over NumPy with C-JIT and 4.76x speedup over NumPy on NVIDIA RTX 5070 GPU.
- **Real Convolutional Neural Network Operations**:
 - Vectorized 2D spatial convolution (`Conv2D`) forward and backward pass with image sliding window view and tensordot.
 - Spatial max pooling (`MaxPooling2D`) forward pass and backward pass with spatial boolean max masks.
- **Sequential Model Framework Bridges**:
 - `model.to_pytorch()` converting NeuroGebra `Model` into PyTorch `nn.Sequential` with full layer weight migration.
 - `model.to_tensorflow()` converting NeuroGebra `Model` into TensorFlow `tf.keras.Sequential`.
- **Autograd Engine Overhaul**:
 - Gradient unbroadcasting (`_unbroadcast`) in `Tensor` preventing broadcasting shape crashes.
 - Topological sort reverse-mode execution in `Tensor.backward()`.
 - Added matrix multiplication (`__matmul__` / `@`), `.T`, `transpose()`, `reshape()`, `relu()`, `sigmoid()`, and `tanh()`.

### Changed

- Replaced simulated fake training curves in `EducationalTrainer` with authentic forward passes, real losses, analytical backpropagation, and parameter optimization.
- Replaced mock random outputs in `Model.predict()` with real inference evaluation.
- Fixed Adam optimizer step counter in `Trainer` so bias correction factors decay once per batch step.
- Upgraded dataset loaders with structured mathematical manifolds instead of pure random noise.
- Cleaned and formatted entire codebase to 100% Ruff and PEP 8 compliance (0 errors, 0 warnings).

## [2.5.11] - 2026-09-01

### Changed

- Replaced the NeuroGebra logo with the refreshed teal brand asset
- Streamlined the README badge header around package, compatibility, CI, coverage, documentation, licensing, and lifetime-download signals
- Updated GitHub Actions to Node.js 24-compatible major versions
- Aligned the publishing workflow with the configured PyPI repository secret names

### Fixed

- Made the README logo use an absolute GitHub raw URL so it renders on both GitHub and PyPI

## [2.5.10] - 2026-04-07

### Fixed

- Replaced the unsupported README dashboard iframe with a GitHub-safe dashboard link block

### Changed

- Version bump to 2.5.10 across package metadata and documentation banners

## [2.5.9] - 2026-04-07

### Added

- Added the location-based downloads dashboard to the README

### Changed

- Version bump to 2.5.9 across package metadata and documentation banners

## [2.5.8] - 2026-04-07

### Fixed

- PyPI project description logo rendering by switching README image source to an absolute GitHub raw URL

### Changed

- Version bump to 2.5.8 across package metadata and documentation banners

## [2.5.7] - 2026-04-07

### Added

- 10 intermediate-to-advanced real-dataset project notebooks under `examples/projects/` (image classification, GANs, diffusion, NLP, and small language model workflows)
- New docs index page for project notebooks: `docs/examples/project_notebooks.md`

### Changed

- README and docs updated to surface the new project notebook portfolio
- Bridge docs and runtime validation coverage expanded to reflect current bridge behavior
- Read the Docs configuration consolidated to `.readthedocs.yaml`
- Version bump to 2.5.7 across package metadata and documentation

## [2.5.3] - 2026-03-01

### Changed -- Documentation and Packaging Overhaul

- **Professional README** -- Complete rewrite with expanded badges, feature comparison matrix, end-to-end Observatory training example, and dedicated sections for students, researchers, and engineers
- **Updated documentation** -- Refreshed landing page, Observatory Pro docs updated to reference v2.5.3
- **Version bump** from 1.3.0 to 2.5.3

### Highlights
- 12 unique features documented with code examples
- Full end-to-end training walkthrough with all Observatory Pro features
- "Who Should Use Neurogebra?" section with tailored guidance for students, researchers, and engineers
- 20+ badges covering CI, coverage, downloads, activity, and more

[2.5.11]: https://github.com/fahiiim/NeuroGebra/releases/tag/v2.5.11
[2.5.10]: https://github.com/fahiiim/NeuroGebra/releases/tag/v2.5.10
[2.5.9]: https://github.com/fahiiim/NeuroGebra/releases/tag/v2.5.9
[2.5.8]: https://github.com/fahiiim/NeuroGebra/releases/tag/v2.5.8
[2.5.7]: https://github.com/fahiiim/NeuroGebra/releases/tag/v2.5.7
[2.5.3]: https://github.com/fahiiim/NeuroGebra/releases/tag/v2.5.3

## [1.3.0] - 2026-02-27

### Added — Observatory Pro 

Six major upgrades that turn the Training Observatory from a passive logger into an **active diagnostic engine**.

#### New Module: `neurogebra.logging.adaptive`
- **`AdaptiveLogger`** — Wraps `TrainingLogger` and stays at BASIC level until an anomaly is detected, then escalates to EXPERT detail automatically (80-90% log reduction)
- **`AnomalyConfig`** — Configurable thresholds for dead neurons, gradient spikes, vanishing/exploding gradients, NaN/Inf, loss spikes, weight stagnation, activation saturation
- **`AnomalyRecord`** — Structured anomaly data with type, severity, layer name, values, and human-readable message

#### New Module: `neurogebra.logging.health_warnings`
- **`AutoHealthWarnings`** — Threshold-based rule engine that runs on every batch/epoch, emitting structured `HealthWarning` objects with diagnosis and actionable recommendations
- **`WarningConfig`** — Configurable thresholds and patience values for all 10 health rules
- **`HealthWarning`** — Structured alert with rule name, severity, message, diagnosis, recommendations, and context (layer/epoch/batch)
- **10 Built-in Rules** — dead_relu, vanishing/exploding gradient, gradient_spike, nan_inf_loss, overfitting, loss_stagnation, weight_stagnation, loss_divergence, activation_saturation

#### New Module: `neurogebra.logging.epoch_summary`
- **`EpochSummarizer`** — Aggregates batch-level metrics and produces per-epoch statistical summaries
- **`EpochSummary`** — Full epoch summary with batch count, duration, and per-metric stats
- **`EpochStats`** — Mean, std, min, max, first, last for each tracked metric
- Human-readable `format_text()` and machine-readable `to_dict()` output

#### New Module: `neurogebra.logging.tiered_storage`
- **`TieredStorage`** — Splits log events into 3 NDJSON files: `basic.log`, `health.log`, `debug.log`
- Buffered writes with configurable flush interval
- `write_debug` flag to disable debug tier in production
- Event classification by type and severity

#### New Module: `neurogebra.logging.dashboard`
- **`DashboardExporter`** — Generates self-contained interactive HTML dashboard with Chart.js (loss curves, accuracy curves, timing bar chart, batch loss, health diagnostics)
- **`TensorBoardBridge`** — Optional TensorBoard integration via `torch.utils.tensorboard`
- **`WandBBridge`** — Optional Weights & Biases integration

#### New Module: `neurogebra.logging.fingerprint`
- **`TrainingFingerprint`** — Captures everything needed to reproduce a training run: seeds, dataset hash, versions, hardware info, OS, model architecture hash, hyperparameters, git state
- `capture()` classmethod for one-line fingerprint creation
- `to_dict()` / `from_dict()` for JSON serialisation
- `format_text()` for human-readable output

### Changed
- Version bumped from 1.2.1 to 1.3.0
- Updated `__init__.py` to export all Observatory Pro classes
- Updated main package docstring to reference Observatory Pro v1.3.0

### Tests
- Added `tests/test_logging/test_observatory_pro.py` — 56 comprehensive tests covering all 6 new features
- Total test count: 470 (all passing)

### Documentation
- Added `docs/advanced/observatory-pro.md` — Full documentation for all 6 features with usage examples and API reference
- Updated `mkdocs.yml` navigation with Observatory Pro entry

[1.3.0]: https://github.com/fahiiim/NeuroGebra/releases/tag/v1.3.0

## [1.2.1] - 2026-02-16

### Added — Training Observatory 

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