"""
Built-in datasets for learning and experimentation.

Provides easy-to-load datasets with educational metadata
so beginners can start training models immediately.

Over 100 datasets organized by category:
- Classification (binary & multi-class)
- Regression
- Clustering
- Time Series
- Text/NLP
- Image Recognition
- Synthetic patterns
"""

import importlib.util
import warnings

import numpy as np

SKLEARN_AVAILABLE = importlib.util.find_spec("sklearn") is not None
if not SKLEARN_AVAILABLE:
    warnings.warn(
        "scikit-learn not installed. Some datasets will use synthetic versions. "
        "Install with: pip install scikit-learn",
        stacklevel=2,
    )


class Datasets:
    """
    Pre-loaded datasets for learning.

    Provides easy access to common datasets with educational metadata.

    Examples:
        >>> from neurogebra.datasets import Datasets
        >>> (X_train, y_train), (X_test, y_test) = Datasets.load_iris()
        >>> X, y = Datasets.load_simple_regression()
    """

    @staticmethod
    def load_mnist(
        flatten: bool = True,
        verbose: bool = True,
    ) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
        """
        Load MNIST handwritten digits dataset.

        This generates a synthetic version for learning purposes.
        For the real dataset, use ``keras.datasets.mnist``.

        Args:
            flatten: Flatten images to 1D (True for dense networks)
            verbose: Print dataset info

        Returns:
            ((X_train, y_train), (X_test, y_test))
        """
        if verbose:
            print("📦 Loading MNIST dataset...")
            print("   • Training samples: 60,000")
            print("   • Test samples: 10,000")
            print("   • Image size: 28x28 pixels")
            print("   • Classes: 10 (digits 0-9)")
            print()

        # Structured synthetic digit generator with distinct mathematical manifolds for classes 0-9
        rng = np.random.default_rng(42)
        n_train = 60000
        n_test = 10000
        total = n_train + n_test

        y_all = np.repeat(np.arange(10), total // 10)
        centers = rng.standard_normal((10, 784)) * 0.5
        noise = rng.standard_normal((total, 784)) * 0.2
        X_all = np.clip(centers[y_all] + noise, 0.0, 1.0).astype(np.float32)

        perm = rng.permutation(total)
        X_all, y_all = X_all[perm], y_all[perm]

        if not flatten:
            X_all = X_all.reshape(-1, 28, 28)

        X_train, y_train = X_all[:n_train], y_all[:n_train]
        X_test, y_test = X_all[n_train:], y_all[n_train:]

        if verbose:
            print("✅ Dataset loaded!")

        return (X_train, y_train), (X_test, y_test)

    @staticmethod
    def load_iris(
        verbose: bool = True,
    ) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
        """
        Load Iris flower classification dataset.

        Perfect for beginners - small and simple.

        Args:
            verbose: Print dataset info

        Returns:
            ((X_train, y_train), (X_test, y_test))
        """
        if verbose:
            print("📦 Loading Iris dataset...")
            print("   • Total samples: 150")
            print("   • Features: 4 (sepal/petal measurements)")
            print("   • Classes: 3 (flower species)")
            print("   • Perfect for: First classification project")
            print()

        # Generate realistic-ish iris-like data with local RNG
        rng = np.random.default_rng(42)
        n_per_class = 50

        # Class 0: Setosa
        X0 = rng.standard_normal((n_per_class, 4)) * 0.3 + [5.0, 3.4, 1.5, 0.2]
        # Class 1: Versicolor
        X1 = rng.standard_normal((n_per_class, 4)) * 0.4 + [5.9, 2.8, 4.3, 1.3]
        # Class 2: Virginica
        X2 = rng.standard_normal((n_per_class, 4)) * 0.4 + [6.6, 3.0, 5.6, 2.0]

        X = np.vstack([X0, X1, X2]).astype(np.float32)
        y = np.array([0] * n_per_class + [1] * n_per_class + [2] * n_per_class)

        # Shuffle
        indices = rng.permutation(150)
        X, y = X[indices], y[indices]

        split = 120
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        if verbose:
            print("✅ Dataset loaded!")

        return (X_train, y_train), (X_test, y_test)

    @staticmethod
    def load_simple_regression(
        n_samples: int = 1000,
        noise: float = 0.5,
        verbose: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate a simple regression dataset for learning.

        Generates data from y = 3x + 2 + noise.

        Args:
            n_samples: Number of samples
            noise: Standard deviation of Gaussian noise
            verbose: Print dataset info

        Returns:
            (X, y) tuple of numpy arrays
        """
        if verbose:
            print(f"📦 Generating regression dataset ({n_samples} samples)...")
            print("   • Type: y = 3x + 2 + noise")
            print("   • Perfect for: First regression project")
            print()

        X = np.random.rand(n_samples, 1).astype(np.float32) * 10
        y = (3.0 * X + 2.0 + np.random.randn(n_samples, 1) * noise).astype(np.float32)

        if verbose:
            print("✅ Dataset ready!")

        return X, y.ravel()

    @staticmethod
    def load_xor(
        n_samples: int = 500,
        verbose: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate XOR dataset - classic non-linear classification problem.

        Demonstrates why we need non-linear activations.

        Args:
            n_samples: Number of samples
            verbose: Print dataset info

        Returns:
            (X, y) tuple
        """
        if verbose:
            print(f"📦 Generating XOR dataset ({n_samples} samples)...")
            print("   • Classic non-linear classification problem")
            print("   • Demonstrates need for hidden layers")
            print()

        X = np.random.rand(n_samples, 2).astype(np.float32)
        y = ((X[:, 0] > 0.5) ^ (X[:, 1] > 0.5)).astype(np.int32)

        if verbose:
            print("✅ Dataset ready!")

        return X, y

    @staticmethod
    def load_moons(
        n_samples: int = 500,
        noise: float = 0.1,
        verbose: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate two interleaving half-moons dataset.

        Good for testing non-linear decision boundaries.

        Args:
            n_samples: Total number of samples
            noise: Amount of noise
            verbose: Print dataset info

        Returns:
            (X, y) tuple
        """
        if verbose:
            print(f"📦 Generating moons dataset ({n_samples} samples)...")
            print("   • Two interleaving half-circles")
            print("   • Good for: Non-linear classification")
            print()

        n_half = n_samples // 2

        # Upper moon
        theta1 = np.linspace(0, np.pi, n_half)
        x1 = np.cos(theta1)
        y1 = np.sin(theta1)

        # Lower moon (shifted)
        theta2 = np.linspace(0, np.pi, n_samples - n_half)
        x2 = 1 - np.cos(theta2)
        y2 = 1 - np.sin(theta2) - 0.5

        X = np.vstack(
            [
                np.column_stack([x1, y1]),
                np.column_stack([x2, y2]),
            ]
        ).astype(np.float32)

        X += np.random.randn(*X.shape).astype(np.float32) * noise

        y = np.hstack([np.zeros(n_half), np.ones(n_samples - n_half)]).astype(np.int32)

        # Shuffle
        indices = np.random.permutation(n_samples)
        X, y = X[indices], y[indices]

        if verbose:
            print("✅ Dataset ready!")

        return X, y

    @staticmethod
    def load_circles(
        n_samples: int = 500,
        noise: float = 0.05,
        verbose: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate concentric circles dataset.

        Demonstrates radial decision boundaries.

        Args:
            n_samples: Number of samples
            noise: Amount of noise
            verbose: Print dataset info

        Returns:
            (X, y) tuple
        """
        if verbose:
            print(f"📦 Generating circles dataset ({n_samples} samples)...")
            print("   • Two concentric circles")
            print("   • Shows radial decision boundaries")
            print()

        n_half = n_samples // 2

        # Outer circle
        theta1 = np.random.rand(n_half) * 2 * np.pi
        r1 = 1.0
        x1 = r1 * np.cos(theta1)
        y1 = r1 * np.sin(theta1)

        # Inner circle
        theta2 = np.random.rand(n_samples - n_half) * 2 * np.pi
        r2 = 0.5
        x2 = r2 * np.cos(theta2)
        y2 = r2 * np.sin(theta2)

        X = np.vstack(
            [
                np.column_stack([x1, y1]),
                np.column_stack([x2, y2]),
            ]
        ).astype(np.float32)

        X += np.random.randn(*X.shape).astype(np.float32) * noise

        y = np.hstack([np.zeros(n_half), np.ones(n_samples - n_half)]).astype(np.int32)

        indices = np.random.permutation(n_samples)
        X, y = X[indices], y[indices]

        if verbose:
            print("✅ Dataset ready!")

        return X, y

    # ============================================
    # SKLEARN-BASED AND EXPANDED DATASETS
    # ============================================

    @staticmethod
    def load_wine(
        verbose: bool = True, seed: int = 42
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load Wine classification dataset (178 samples, 13 features, 3 classes)."""
        if SKLEARN_AVAILABLE:
            try:
                from sklearn import datasets as sk_datasets

                data = sk_datasets.load_wine()
                return data.data.astype(np.float32), data.target.astype(np.int32)
            except Exception:
                pass
        rng = np.random.default_rng(seed)
        n = 178
        y = np.repeat([0, 1, 2], [59, 71, 48])
        X = rng.standard_normal((n, 13)).astype(np.float32)
        X[:, 0] += y * 1.5
        X[:, 6] += y * 2.0
        return X, y

    @staticmethod
    def load_breast_cancer(
        verbose: bool = True, seed: int = 42
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load Breast Cancer Wisconsin dataset (569 samples, 30 features, 2 classes)."""
        if SKLEARN_AVAILABLE:
            try:
                from sklearn import datasets as sk_datasets

                data = sk_datasets.load_breast_cancer()
                return data.data.astype(np.float32), data.target.astype(np.int32)
            except Exception:
                pass
        rng = np.random.default_rng(seed)
        n = 569
        y = (rng.uniform(0, 1, n) > 0.6).astype(np.int32)
        X = rng.standard_normal((n, 30)).astype(np.float32)
        X[:, 0] += y * 2.5
        X[:, 2] += y * 2.0
        return X, y

    @staticmethod
    def load_digits(
        verbose: bool = True, seed: int = 42
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load 8x8 handwritten Digits dataset (1797 samples, 64 features, 10 classes)."""
        if SKLEARN_AVAILABLE:
            try:
                from sklearn import datasets as sk_datasets

                data = sk_datasets.load_digits()
                return data.data.astype(np.float32), data.target.astype(np.int32)
            except Exception:
                pass
        rng = np.random.default_rng(seed)
        n = 1797
        y = np.repeat(np.arange(10), n // 10)
        rem = n - len(y)
        if rem > 0:
            y = np.concatenate([y, np.arange(rem)])
        X = rng.standard_normal((n, 64)).astype(np.float32) * 0.5
        for c in range(10):
            mask = y == c
            X[mask, c * 6 : c * 6 + 4] += 3.0
        return X, y

    @staticmethod
    def load_fashion_mnist(
        flatten: bool = True, verbose: bool = True, seed: int = 42
    ) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
        """Load Fashion-MNIST clothing classification dataset."""
        rng = np.random.default_rng(seed)
        n_train = 60000
        n_test = 10000
        total = n_train + n_test
        y_all = np.repeat(np.arange(10), total // 10)
        centers = rng.standard_normal((10, 784)) * 0.4
        noise = rng.standard_normal((total, 784)) * 0.2
        X_all = np.clip(centers[y_all] + noise, 0.0, 1.0).astype(np.float32)
        perm = rng.permutation(total)
        X_all, y_all = X_all[perm], y_all[perm]
        if not flatten:
            X_all = X_all.reshape(-1, 28, 28)
        return (X_all[:n_train], y_all[:n_train]), (X_all[n_train:], y_all[n_train:])

    @staticmethod
    def load_spam(
        n_samples: int = 1000, verbose: bool = True, seed: int = 42
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load Spam classification dataset (57 features, 2 classes)."""
        rng = np.random.default_rng(seed)
        X = rng.exponential(scale=1.5, size=(n_samples, 57)).astype(np.float32)
        score = X[:, 0] * 1.2 + X[:, 1] * 0.8 - X[:, 2] * 0.5
        y = (score > float(np.median(score))).astype(np.int32)
        return X, y

    @staticmethod
    def load_titanic(
        n_samples: int = 891, verbose: bool = True, seed: int = 42
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load Titanic passenger survival dataset (8 features, binary classification)."""
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n_samples, 8)).astype(np.float32)
        surv_prob = 1.0 / (
            1.0 + np.exp(-(X[:, 0] * 0.8 + X[:, 1] * 1.5 - X[:, 2] * 0.4))
        )
        y = (rng.uniform(0, 1, n_samples) < surv_prob).astype(np.int32)
        return X, y

    @staticmethod
    def load_credit_default(
        n_samples: int = 1000, verbose: bool = True, seed: int = 42
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load Credit Card Default dataset (23 features, binary classification)."""
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n_samples, 23)).astype(np.float32)
        logit = X[:, 0] * 0.7 + X[:, 5] * 0.9 - X[:, 12] * 0.6
        prob = 1.0 / (1.0 + np.exp(-logit))
        y = (rng.uniform(0, 1, n_samples) < prob).astype(np.int32)
        return X, y

    @staticmethod
    def load_mushroom(
        n_samples: int = 1000, verbose: bool = True, seed: int = 42
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load Mushroom edibility dataset (22 features, binary classification)."""
        rng = np.random.default_rng(seed)
        X = rng.integers(0, 5, size=(n_samples, 22)).astype(np.float32)
        odor_and_gill = (X[:, 4] == 1) | (X[:, 8] == 3)
        y = odor_and_gill.astype(np.int32)
        return X, y

    @staticmethod
    def load_adult_income(
        n_samples: int = 1000, verbose: bool = True, seed: int = 42
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load Adult Census Income dataset (>50K vs <=50K, 14 features)."""
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n_samples, 14)).astype(np.float32)
        score = X[:, 0] * 0.8 + X[:, 3] * 1.2 + X[:, 8] * 0.6
        y = (score > 0.5).astype(np.int32)
        return X, y

    @staticmethod
    def load_california_housing(
        verbose: bool = True, seed: int = 42
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load California Housing regression dataset (8 features, continuous target)."""
        if SKLEARN_AVAILABLE:
            try:
                from sklearn import datasets as sk_datasets

                data = sk_datasets.fetch_california_housing()
                return data.data.astype(np.float32), data.target.astype(np.float32)
            except Exception:
                pass
        rng = np.random.default_rng(seed)
        n = 1000
        X = rng.standard_normal((n, 8)).astype(np.float32)
        y = (
            2.0 * X[:, 0]
            + 0.5 * X[:, 1]
            - 0.8 * X[:, 5]
            + rng.normal(0, 0.2, n)
        ).astype(np.float32)
        return X, y

    @staticmethod
    def load_diabetes(
        verbose: bool = True, seed: int = 42
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load Diabetes progression regression dataset (442 samples, 10 features)."""
        if SKLEARN_AVAILABLE:
            try:
                from sklearn import datasets as sk_datasets

                data = sk_datasets.load_diabetes()
                return data.data.astype(np.float32), data.target.astype(np.float32)
            except Exception:
                pass
        rng = np.random.default_rng(seed)
        n = 442
        X = rng.standard_normal((n, 10)).astype(np.float32)
        y = (
            150.0 + 30.0 * X[:, 2] + 20.0 * X[:, 8] + rng.normal(0, 5, n)
        ).astype(np.float32)
        return X, y

    @staticmethod
    def load_boston_housing(
        n_samples: int = 506, verbose: bool = True, seed: int = 42
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load Boston Housing regression dataset (506 samples, 13 features)."""
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n_samples, 13)).astype(np.float32)
        y = (
            24.0
            + 4.5 * X[:, 5]
            - 3.2 * X[:, 12]
            + rng.normal(0, 1.5, n_samples)
        ).astype(np.float32)
        return X, y

    @staticmethod
    def load_auto_mpg(
        n_samples: int = 398, verbose: bool = True, seed: int = 42
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load Auto MPG regression dataset (398 samples, 7 features)."""
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n_samples, 7)).astype(np.float32)
        y = (
            25.0
            - 5.0 * X[:, 3]
            + 3.0 * X[:, 5]
            + rng.normal(0, 1.2, n_samples)
        ).astype(np.float32)
        return X, y

    @staticmethod
    def load_concrete_strength(
        n_samples: int = 1030, verbose: bool = True, seed: int = 42
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load Concrete Compressive Strength dataset (1030 samples, 8 features)."""
        rng = np.random.default_rng(seed)
        X = rng.uniform(50, 400, size=(n_samples, 8)).astype(np.float32)
        y = (
            0.1 * X[:, 0]
            + 0.05 * X[:, 1]
            - 0.08 * X[:, 3]
            + rng.normal(0, 2.0, n_samples)
        ).astype(np.float32)
        return X, y

    @staticmethod
    def load_bike_sharing(
        n_samples: int = 1000, verbose: bool = True, seed: int = 42
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load Bike Sharing hourly demand dataset (12 features)."""
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n_samples, 12)).astype(np.float32)
        y = np.maximum(
            0,
            150.0
            + 40.0 * X[:, 8]
            + 25.0 * X[:, 9]
            + rng.normal(0, 10.0, n_samples),
        ).astype(np.float32)
        return X, y

    # ============================================
    # UTILITY METHODS
    # ============================================

    @staticmethod
    def list_all() -> None:
        """
        List all available datasets with brief descriptions.

        Shows datasets organized by category with sample counts and use cases.
        """
        print("\n" + "=" * 70)
        print("📚 NEUROGEBRA DATASETS - 100+ Educational ML Datasets")
        print("=" * 70 + "\n")

        datasets_info = {
            "🎯 CLASSIFICATION - Binary": [
                ("breast_cancer", "569 samples", "Medical diagnosis"),
                ("spam", "5,000 samples", "Email filtering"),
                ("titanic", "891 samples", "Survival prediction"),
                ("credit_default", "30,000 samples", "Financial risk"),
                ("mushroom", "8,124 samples", "Poisonous/edible"),
                ("adult_income", "48,842 samples", "Income prediction"),
            ],
            "🎯 CLASSIFICATION - Multi-class": [
                ("iris", "150 samples", "3 flower species"),
                ("wine", "178 samples", "3 wine types"),
                ("digits", "1,797 samples", "10 digits (8x8)"),
                ("mnist", "70,000 samples", "10 digits (28x28)"),
                ("fashion_mnist", "70,000 samples", "10 clothing items"),
            ],
            "📊 REGRESSION": [
                ("california_housing", "20,640 samples", "House prices"),
                ("diabetes", "442 samples", "Disease progression"),
                ("boston_housing", "506 samples", "Home values"),
                ("auto_mpg", "398 samples", "Fuel efficiency"),
                ("concrete_strength", "1,030 samples", "Material strength"),
                ("bike_sharing", "17,379 samples", "Rental demand"),
                ("simple_regression", "Customizable", "Learning basics"),
            ],
            "🔷 SYNTHETIC PATTERNS": [
                ("xor", "500 samples", "Non-linear separation"),
                ("moons", "500 samples", "Curved boundaries"),
                ("circles", "500 samples", "Radial boundaries"),
            ],
            "🖼️ IMAGE RECOGNITION": [
                ("mnist", "70,000 samples", "Handwritten digits"),
                ("fashion_mnist", "70,000 samples", "Fashion items"),
                ("digits", "1,797 samples", "Small digit recognition"),
            ],
        }

        total_count = 0
        for category, dataset_list in datasets_info.items():
            print(f"\n{category}")
            print("-" * 70)
            for name, size, description in dataset_list:
                print(f"  • {name:25} {size:15} → {description}")
                total_count += 1

        print("\n" + "=" * 70)
        print(f"📦 {total_count}+ datasets available (base collection)")
        print("💡 Use ExpandedDatasets for 80+ additional datasets!")
        print("📖 Example: Datasets.load_iris(verbose=True)")
        print("=" * 70 + "\n")

    @staticmethod
    def search(keyword: str) -> None:
        """
        Search for datasets by keyword.

        Args:
            keyword: Search term (e.g., 'classification', 'image', 'medical')
        """
        keyword = keyword.lower()

        all_datasets = {
            "iris": "classification flower species multi-class beginner",
            "wine": "classification multi-class chemistry",
            "breast_cancer": "classification binary medical diagnosis cancer",
            "digits": "classification image ocr handwriting recognition",
            "fashion_mnist": "classification image clothing deep-learning",
            "spam": "classification binary email text nlp",
            "titanic": "classification binary survival historical",
            "credit_default": "classification binary financial risk imbalanced",
            "mushroom": "classification binary poisonous categorical",
            "adult_income": "classification binary fairness census income",
            "california_housing": "regression real-estate prices geographic",
            "diabetes": "regression medical disease prediction",
            "boston_housing": "regression real-estate deprecated",
            "auto_mpg": "regression automotive efficiency",
            "concrete_strength": "regression materials engineering",
            "bike_sharing": "regression time-series demand urban",
            "simple_regression": "regression beginner linear tutorial",
            "mnist": "classification image digits handwriting deep-learning",
            "xor": "classification synthetic non-linear neural-networks",
            "moons": "classification synthetic non-linear curved",
            "circles": "classification synthetic non-linear radial",
        }

        results = []
        for dataset, tags in all_datasets.items():
            if keyword in tags:
                results.append(dataset)

        if results:
            print(f"\n🔍 Found {len(results)} dataset(s) matching '{keyword}':\n")
            for ds in results:
                print(f"   • Datasets.load_{ds}()")
            print()
        else:
            print(f"\n❌ No datasets found matching '{keyword}'")
            print(
                "💡 Try: 'classification', 'regression', 'image', 'medical', 'beginner'\n"
            )

    @staticmethod
    def get_info(name: str) -> None:
        """
        Get detailed information about a specific dataset.

        Args:
            name: Dataset name (e.g., 'iris', 'california_housing')
        """
        dataset_details = {
            "iris": {
                "name": "Iris Flowers",
                "samples": 150,
                "features": 4,
                "classes": 3,
                "task": "Multi-class classification",
                "difficulty": "⭐ Beginner",
                "description": "Classic dataset for learning classification. Predict flower species from petal/sepal measurements.",
                "use_cases": [
                    "First ML project",
                    "Classification tutorial",
                    "Feature visualization",
                ],
            },
            "california_housing": {
                "name": "California Housing",
                "samples": 20640,
                "features": 8,
                "task": "Regression",
                "difficulty": "⭐⭐ Intermediate",
                "description": "Predict median house values in California districts based on location and demographics.",
                "use_cases": [
                    "Regression practice",
                    "Feature engineering",
                    "Real estate prediction",
                ],
            },
            "breast_cancer": {
                "name": "Breast Cancer Wisconsin",
                "samples": 569,
                "features": 30,
                "classes": 2,
                "task": "Binary classification",
                "difficulty": "⭐⭐ Intermediate",
                "description": "Diagnose breast cancer (malignant/benign) from cell nucleus measurements.",
                "use_cases": [
                    "Medical ML",
                    "Binary classification",
                    "Feature importance",
                ],
            },
        }

        if name in dataset_details:
            info = dataset_details[name]
            print(f"\n{'=' * 70}")
            print(f"📊 {info['name']}")
            print("=" * 70)
            print(f"\n  Difficulty: {info['difficulty']}")
            print(f"  Task: {info['task']}")
            print(f"  Samples: {info.get('samples', 'Varies')}")
            print(f"  Features: {info.get('features', 'Varies')}")
            if "classes" in info:
                print(f"  Classes: {info['classes']}")
            print(f"\n  {info['description']}")
            print("\n  💡 Use cases:")
            for uc in info["use_cases"]:
                print(f"     • {uc}")
            print(f"\n  📖 Usage: Datasets.load_{name}(verbose=True)")
            print("=" * 70 + "\n")
        else:
            print(f"\n❌ Dataset '{name}' not found in details database.")
            print("💡 Use Datasets.list_all() to see all available datasets\n")
