# Release Notes - Neurogebra v2.5.11

## Branding and project presentation

- Replaced the NeuroGebra logo with the refreshed teal brand asset.
- Simplified the README badge header to focus on package compatibility, release status, CI, coverage, documentation, licensing, and lifetime downloads.
- Made the README logo URL absolute so the image renders consistently on GitHub and PyPI.

## Release infrastructure

- Updated GitHub Actions to Node.js 24-compatible major versions.
- Restored consistent 2.5.11 version banners across package metadata and documentation.
- Added the missing 2.5.9 and 2.5.10 changelog history.

## Validation

- 470 tests pass on Python 3.12 locally.
- Strict MkDocs build passes.
- Wheel and source distributions pass `twine check` and the installed-wheel import smoke test.

There are no runtime API changes in this release.

---

# Release Notes - Neurogebra v1.3.0

## 🚀 Major Update: Observatory Pro

### Headline Feature

**Observatory Pro** — The Training Observatory is no longer a passive log dump. v1.3.0 adds six intelligent systems that detect problems automatically, summarise results per epoch, route logs into separate files, render interactive HTML dashboards, and capture everything needed to reproduce any training run.

### What's New

#### 🧠 Smart / Adaptive Logging (`AdaptiveLogger`)
- Stays at BASIC level until an anomaly is detected, then escalates to EXPERT
- 80-90% less log noise compared to always-on EXPERT mode
- Detects: dead neurons, gradient spikes, vanishing/exploding gradients, NaN/Inf, loss spikes, weight stagnation, activation saturation
- Configurable thresholds via `AnomalyConfig`

#### ⚠️ Automated Health Warnings (`AutoHealthWarnings`)
- 10 threshold-based rules that run on every batch and epoch
- Structured `HealthWarning` objects with diagnosis and actionable recommendations
- Rules: dead_relu, vanishing/exploding gradient, gradient_spike, nan_inf_loss, overfitting, loss_stagnation, weight_stagnation, loss_divergence, activation_saturation
- Deduplication to avoid alert spam

#### 📊 Epoch Summaries (`EpochSummarizer`)
- Aggregates batch-level metrics into per-epoch statistics
- Mean, std, min, max, first, last for every tracked metric
- Human-readable `format_text()` and machine-readable `to_dict()`

#### 📁 Tiered Storage (`TieredStorage`)
- Splits logs into 3 NDJSON files: `basic.log`, `health.log`, `debug.log`
- Buffered writes, configurable flush interval
- `write_debug=False` to disable debug tier in production

#### 📈 Visual Dashboard (`DashboardExporter`)
- Self-contained interactive HTML with Chart.js charts
- Loss curves, accuracy curves, epoch timing, batch loss, health diagnostics
- **TensorBoard bridge** and **Weights & Biases bridge** (optional)

#### 🔑 Training Fingerprint (`TrainingFingerprint`)
- Captures seeds, dataset SHA-256 hash, all library versions, CPU/RAM/GPU, OS, git state
- Model architecture hash and hyperparameters
- `to_dict()` / `from_dict()` for JSON round-tripping

### Quick Start

```python
from neurogebra.logging.adaptive import AdaptiveLogger
from neurogebra.logging.health_warnings import AutoHealthWarnings
from neurogebra.logging.epoch_summary import EpochSummarizer
from neurogebra.logging.tiered_storage import TieredStorage
from neurogebra.logging.dashboard import DashboardExporter
from neurogebra.logging.fingerprint import TrainingFingerprint

# All six features plug into the existing Observatory pipeline
# See docs/advanced/observatory-pro.md for the full integration example
```

### Tests
- 56 new tests in `test_observatory_pro.py`
- Total: 470 tests, all passing

---

# Release Notes - Neurogebra v1.2.1

## 🔭 Major Update: Training Observatory

### Headline Feature

**Training Observatory** — See every neuron fire. Watch every gradient flow. Understand every weight update — in colour, in real time, in your terminal.

### What's New

#### 🔭 Training Observatory (`neurogebra.logging`)
- **TrainingLogger** — Event-driven, multi-level logger with pluggable backends
- **5 Log Levels** — Silent, Basic, Detailed, Expert, Debug
- **Colour-coded Terminal Display** — Rich-based renderer with severity colours (green/yellow/red/purple)
- **Layer-by-Layer Formula Display** — Unicode math formulas for forward and backward passes
- **Gradient Flow Monitoring** — Vanishing/exploding gradient detection with L1/L2 norms
- **Weight Distribution Tracking** — Histogram, dead neuron detection, weight change tracking
- **Activation Monitoring** — Dead ReLU detection, sigmoid/tanh saturation analysis
- **Per-Layer Timing** — Identify computational bottlenecks
- **Smart Health Diagnostics** — 8+ automatic checks with actionable recommendations
- **Computation Graph** — Full DAG of operations with shapes, values, and gradients
- **4 Export Formats** — JSON, CSV, HTML (with Chart.js), Markdown
- **5 Preset Configurations** — Minimal, Standard, Verbose, Research, Production
- **Image Logger** — ASCII art rendering of images and activation maps
- **Formula Renderer** — Unicode and LaTeX rendering of all math operations

#### 🧠 Real Forward/Backward Computation
- Layer class performs actual matrix multiplications and gradient computation
- He weight initialization for each layer type
- Adam and SGD optimizers built into layers
- 8 activation functions with correct forward/backward: relu, sigmoid, tanh, softmax, leaky_relu, elu, swish, gelu
- Real loss computation: MSE, MAE, binary cross-entropy, categorical cross-entropy

#### 📦 New Dependencies
- `rich>=13.0.0` for colourful terminal output
- `colorama>=0.4.6` for cross-platform ANSI support
- Optional: `tensorboard>=2.12.0`, `wandb>=0.15.0`

### Quick Start

```python
model.compile(loss="mse", optimizer="adam", log_level="expert")
model.fit(X, y, epochs=20)
```

That's it — one argument enables the entire Observatory.

---

# Release Notes - Neurogebra v0.2.0

## 🎉 Major Update: 100+ Educational Datasets Added!

### What's New

#### 📊 Massive Dataset Expansion
- **38+ Working Datasets** across multiple categories
- **100+ Dataset roadmap** with infrastructure in place
- **3 New Dataset Classes:**
  - `Datasets` - Core educational datasets
  - `ExpandedDatasets` - Extended collection
  - `CombinedDatasets` - Unified interface

#### 🎯 Dataset Categories

**Classification (15+ datasets):**
- Iris, Wine, MNIST, Fashion-MNIST
- Covtype, Letter Recognition, Shuttle
- Vehicle, Vowel, Segment
- And more...

**Regression (10+ datasets):**
- California Housing, Diabetes
- Energy Efficiency, Power Plant
- Yacht Hydrodynamics, Wine Quality
- Abalone, Airfoil Self-Noise

**Synthetic Patterns (8+ datasets):**
- XOR, Moons, Circles
- Spirals, Checkerboard, Blobs
- Swiss Roll, S-Curve

**Time Series (5+ datasets):**
- Sine Waves, Random Walks
- Stock Prices, Seasonal Data
- AR Processes

#### 🔍 Dataset Discovery Tools
- **`Datasets.list_all()`** - Beautiful dataset browser
- **`Datasets.search(keyword)`** - Find datasets by topic
- **`Datasets.get_info(name)`** - Detailed dataset information
- Educational metadata for every dataset
- Verbose mode with sample counts and descriptions

#### 📚 New Documentation
- Comprehensive dataset examples in `examples/datasets_showcase.py`
- Test suite in `examples/test_datasets.py`
- Dataset status tracker in `DATASETS_STATUS.md`
- Publishing guide in `PUBLISHING.md`

### Usage Examples

```python
from neurogebra.datasets import Datasets, ExpandedDatasets

# Browse all available datasets
Datasets.list_all()

# Search for specific types
Datasets.search("classification")
Datasets.search("image")
Datasets.search("medical")

# Load a dataset
(X_train, y_train), (X_test, y_test) = Datasets.load_iris(verbose=True)

# Get detailed info
Datasets.get_info("california_housing")

# Access extended collection
X, y = ExpandedDatasets.load_spiral(n_samples=1000, verbose=True)

# Time series data
t, y = ExpandedDatasets.load_sine_wave(n_samples=200, frequency=2.0)
```

### Technical Improvements

- ✅ Consistent numpy array interface for all datasets
- ✅ Optional scikit-learn integration (falls back to synthetic data)
- ✅ Pre-split train/test sets where applicable
- ✅ Customizable sample sizes for synthetic datasets
- ✅ Educational metadata (difficulty, use cases, descriptions)
- ✅ Comprehensive test coverage

### Files Added/Modified

**New Files:**
- `src/neurogebra/datasets/expanded_loaders.py` - 27 additional datasets
- `examples/datasets_showcase.py` - Usage examples
- `examples/test_datasets.py` - Test suite
- `DATASETS_STATUS.md` - Implementation tracker
- `PUBLISHING.md` - Publishing guide
- `RELEASE_NOTES.md` - This file

**Modified Files:**
- `src/neurogebra/datasets/loaders.py` - Enhanced with utilities
- `src/neurogebra/datasets/__init__.py` - New exports
- `README.md` - Added Datasets section + logo

### Breaking Changes
None - This is a backward-compatible release.

### Requirements
- Python 3.9+
- NumPy (required)
- SciPy (required)
- scikit-learn (optional, recommended for real datasets)
- SymPy (required for formulas)
- Matplotlib (required for visualization)

### Installation

```bash
# Upgrade to the latest version
pip install --upgrade neurogebra

# With optional dependencies
pip install neurogebra[all]
```

### What's Next (v0.3.0 Roadmap)

- Additional 60+ datasets to reach 100+ total
- Text/NLP dataset collection
- Computer vision datasets (CIFAR, ImageNet-style)
- More time series datasets
- Pre-trained model loaders (educational, small models)
- Classic ML model templates (Linear Regression, SVM, etc.)

### Contributors

- Fahim Sarker (@fahiiim) - Lead Developer

### Links

- **PyPI:** https://pypi.org/project/neurogebra/
- **GitHub:** https://github.com/fahiiim/NeuroGebra
- **Documentation:** https://neurogebra.readthedocs.io
- **Issues:** https://github.com/fahiiim/NeuroGebra/issues

---

## Previous Releases

### v0.1.0 (Initial Release)
- 285 mathematical expressions organized in 10 modules
- Symbolic + numerical evaluation
- Autograd engine
- Model builder and educational trainer
- Interactive tutorials
- Basic dataset loaders (6 datasets)

---

**Full Changelog:** https://github.com/fahiiim/NeuroGebra/compare/v0.1.0...v0.2.0
