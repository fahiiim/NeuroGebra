# Dataset catalog

**48 unique loaders**, generated from `CombinedDatasets` by `python scripts/generate_reference.py`.

Most loaders generate synthetic teaching data, including the MNIST, Iris, Fashion-MNIST, Titanic and other familiar names. These are not downloads of the original benchmark datasets. Five loaders optionally use scikit-learn sources and fall back to synthetic data. California housing may require a network fetch when using scikit-learn. Record which source was used before reporting accuracy.

Return conventions are part of each loader's API. **Pair** means `(X, y)`; **split** means `((X_train, y_train), (X_test, y_test))`. Sizes and options appear below. Large image loaders can allocate substantial memory; start with a small tabular generator.

```python
from neurogebra.datasets import Datasets

# A pair: split it explicitly if you need train/test partitions.
X, y = Datasets.load_digits(verbose=False)

# A split: already partitioned by the loader (synthetic Iris data).
(X_train, y_train), (X_test, y_test) = Datasets.load_iris(verbose=False)
```

## Base loaders

| Loader and options | Result | Provenance |
|---|---|---|
| `load_adult_income(n_samples=1000, verbose=True, seed=42)` | Pair | Synthetic / generated |
| `load_auto_mpg(n_samples=398, verbose=True, seed=42)` | Pair | Synthetic / generated |
| `load_bike_sharing(n_samples=1000, verbose=True, seed=42)` | Pair | Synthetic / generated |
| `load_boston_housing(n_samples=506, verbose=True, seed=42)` | Pair | Synthetic / generated |
| `load_breast_cancer(verbose=True, seed=42)` | Pair | scikit-learn source if available; synthetic fallback |
| `load_california_housing(verbose=True, seed=42)` | Pair | scikit-learn source if available; synthetic fallback |
| `load_circles(n_samples=500, noise=0.05, verbose=True)` | Pair | Synthetic / generated |
| `load_concrete_strength(n_samples=1030, verbose=True, seed=42)` | Pair | Synthetic / generated |
| `load_credit_default(n_samples=1000, verbose=True, seed=42)` | Pair | Synthetic / generated |
| `load_diabetes(verbose=True, seed=42)` | Pair | scikit-learn source if available; synthetic fallback |
| `load_digits(verbose=True, seed=42)` | Pair | scikit-learn source if available; synthetic fallback |
| `load_fashion_mnist(flatten=True, verbose=True, seed=42)` | Split | Synthetic / generated |
| `load_iris(verbose=True)` | Split | Synthetic / generated |
| `load_mnist(flatten=True, verbose=True)` | Split | Synthetic / generated |
| `load_moons(n_samples=500, noise=0.1, verbose=True)` | Pair | Synthetic / generated |
| `load_mushroom(n_samples=1000, verbose=True, seed=42)` | Pair | Synthetic / generated |
| `load_simple_regression(n_samples=1000, noise=0.5, verbose=True)` | Pair | Synthetic / generated |
| `load_spam(n_samples=1000, verbose=True, seed=42)` | Pair | Synthetic / generated |
| `load_titanic(n_samples=891, verbose=True, seed=42)` | Pair | Synthetic / generated |
| `load_wine(verbose=True, seed=42)` | Pair | scikit-learn source if available; synthetic fallback |
| `load_xor(n_samples=500, verbose=True)` | Pair | Synthetic / generated |

## Expanded loaders

| Loader and options | Result | Provenance |
|---|---|---|
| `load_abalone(n_samples=4177, verbose=True)` | Split | Synthetic / generated |
| `load_airfoil_self_noise(n_samples=1503, verbose=True)` | Split | Synthetic / generated |
| `load_ar_process(n_samples=1000, ar_coef=0.8, noise=0.5, verbose=True)` | Pair | Synthetic / generated |
| `load_blobs(n_samples=1000, n_features=2, centers=3, cluster_std=1.0, verbose=True)` | Pair | Synthetic / generated |
| `load_checkerboard(n_samples=1000, verbose=True)` | Pair | Synthetic / generated |
| `load_connect4(n_samples=67557, verbose=True)` | Split | Synthetic / generated |
| `load_covtype(n_samples=10000, verbose=True)` | Split | Synthetic / generated |
| `load_energy_efficiency(n_samples=768, verbose=True)` | Split | Synthetic / generated |
| `load_half_kernel(n_samples=1000, verbose=True)` | Pair | Synthetic / generated |
| `load_letter_recognition(n_samples=20000, verbose=True)` | Split | Synthetic / generated |
| `load_optical_recognition(n_samples=5620, verbose=True)` | Split | Synthetic / generated |
| `load_pendigits(n_samples=10992, verbose=True)` | Split | Synthetic / generated |
| `load_power_plant(n_samples=9568, verbose=True)` | Split | Synthetic / generated |
| `load_random_walk(n_samples=1000, drift=0.0, noise=1.0, verbose=True)` | Pair | Synthetic / generated |
| `load_s_curve(n_samples=1000, noise=0.0, verbose=True)` | Pair | Synthetic / generated |
| `load_satimage(n_samples=6435, verbose=True)` | Split | Synthetic / generated |
| `load_seasonal_data(n_samples=365, period=30, trend=0.01, noise=0.5, verbose=True)` | Pair | Synthetic / generated |
| `load_segment(n_samples=2310, verbose=True)` | Split | Synthetic / generated |
| `load_shuttle(n_samples=58000, verbose=True)` | Split | Synthetic / generated |
| `load_sine_wave(n_samples=1000, frequency=1.0, noise=0.1, verbose=True)` | Pair | Synthetic / generated |
| `load_spiral(n_samples=1000, noise=0.2, verbose=True)` | Pair | Synthetic / generated |
| `load_stock_prices(n_samples=252, starting_price=100.0, volatility=0.02, drift=0.001, verbose=True)` | Pair | Synthetic / generated |
| `load_swiss_roll(n_samples=1500, noise=0.0, verbose=True)` | Pair | Synthetic / generated |
| `load_vehicle(n_samples=846, verbose=True)` | Split | Synthetic / generated |
| `load_vowel(n_samples=990, verbose=True)` | Split | Synthetic / generated |
| `load_wine_quality_red(n_samples=1599, verbose=True)` | Split | Synthetic / generated |
| `load_yacht_hydrodynamics(n_samples=308, verbose=True)` | Split | Synthetic / generated |

## Use original benchmarks

For benchmark comparisons, obtain original data from its documented publisher or a framework's official dataset loader. Preserve the original train/test split, license, preprocessing and attribution. Installing the `datasets` extra adds scikit-learn; it does not turn every synthetic loader into an original-data loader.
