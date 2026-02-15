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
