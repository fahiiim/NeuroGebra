# Datasets

NeuroGebra exposes 48 unique loader methods through `CombinedDatasets`: 21 base loaders and 27 expanded loaders. The [complete catalog](../reference/datasets.md) lists every method, options, return convention and provenance.

## Load a small teaching dataset

```python
from neurogebra.datasets import Datasets

(X_train, y_train), (X_test, y_test) = Datasets.load_iris(verbose=False)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
```

This loader generates Iris-like synthetic data; it does not return the original Iris dataset. Familiar names are teaching conveniences, so always check provenance before reporting a benchmark result.

## Handle pair-returning loaders

```python
from neurogebra.datasets import Datasets

X, y = Datasets.load_digits(verbose=False)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
```

This example illustrates unpacking. For evaluation, shuffle or stratify before splitting; the synthetic digits fallback groups examples by class. Fit preprocessing using only the training partition.

## Use the expanded generators

```python
from neurogebra.datasets import CombinedDatasets

X, y = CombinedDatasets.load_checkerboard(n_samples=200, verbose=False)
print(X.shape, y.shape)
```

Expanded generators currently use NumPy's global RNG in several paths. For strict reproducibility, record seeds and isolate generation from training; conversion to local generator instances is tracked in the [review](../project/review-3.md).

## Original data and optional integrations

Wine, Breast Cancer, Digits, Diabetes and California Housing can use scikit-learn sources when available, with a synthetic fallback on failure. California Housing can require a download. The other loaders generate distributions locally. The `datasets` extra installs scikit-learn but does not replace every synthetic generator with original benchmark data.

For published results, use the original dataset publisher or an official framework loader, record the data license and preprocessing, and preserve an uncontaminated test set.
