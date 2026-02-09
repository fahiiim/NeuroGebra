# Changelog

All notable changes to Neurogebra will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
