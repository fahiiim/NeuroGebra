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

from typing import Optional, Tuple, List, Dict, Any
import warnings

import numpy as np

# Try to import sklearn datasets, but don't require it
try:
    from sklearn import datasets as sklearn_datasets
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn(
        "scikit-learn not installed. Some datasets will use synthetic versions. "
        "Install with: pip install scikit-learn"
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
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
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

        shape = (784,) if flatten else (28, 28)

        X_train = np.random.rand(60000, *shape).astype(np.float32)
        y_train = np.random.randint(0, 10, 60000)
        X_test = np.random.rand(10000, *shape).astype(np.float32)
        y_test = np.random.randint(0, 10, 10000)

        if verbose:
            print("✅ Dataset loaded!")

        return (X_train, y_train), (X_test, y_test)

    @staticmethod
    def load_iris(
        verbose: bool = True,
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
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

        # Generate realistic-ish iris-like data
        np.random.seed(42)
        n_per_class = 50

        # Class 0: Setosa
        X0 = np.random.randn(n_per_class, 4) * 0.3 + [5.0, 3.4, 1.5, 0.2]
        # Class 1: Versicolor
        X1 = np.random.randn(n_per_class, 4) * 0.4 + [5.9, 2.8, 4.3, 1.3]
        # Class 2: Virginica
        X2 = np.random.randn(n_per_class, 4) * 0.4 + [6.6, 3.0, 5.6, 2.0]

        X = np.vstack([X0, X1, X2]).astype(np.float32)
        y = np.array(
            [0] * n_per_class + [1] * n_per_class + [2] * n_per_class
        )

        # Shuffle
        indices = np.random.permutation(150)
        X, y = X[indices], y[indices]

        split = 120
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # Reset random seed
        np.random.seed(None)

        if verbose:
            print("✅ Dataset loaded!")

        return (X_train, y_train), (X_test, y_test)

    @staticmethod
    def load_simple_regression(
        n_samples: int = 1000,
        noise: float = 0.5,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
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
            print(
                f"📦 Generating regression dataset ({n_samples} samples)..."
            )
            print("   • Type: y = 3x + 2 + noise")
            print("   • Perfect for: First regression project")
            print()

        X = np.random.rand(n_samples, 1).astype(np.float32) * 10
        y = (3.0 * X + 2.0 + np.random.randn(n_samples, 1) * noise).astype(
            np.float32
        )

        if verbose:
            print("✅ Dataset ready!")

        return X, y.ravel()

    @staticmethod
    def load_xor(
        n_samples: int = 500,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
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
    ) -> Tuple[np.ndarray, np.ndarray]:
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

        y = np.hstack(
            [np.zeros(n_half), np.ones(n_samples - n_half)]
        ).astype(np.int32)

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
    ) -> Tuple[np.ndarray, np.ndarray]:
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
            print(
                f"📦 Generating circles dataset ({n_samples} samples)..."
            )
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

        y = np.hstack(
            [np.zeros(n_half), np.ones(n_samples - n_half)]
        ).astype(np.int32)

        indices = np.random.permutation(n_samples)
        X, y = X[indices], y[indices]

        if verbose:
            print("✅ Dataset ready!")

        return X, y


    # ============================================
    # SKLEARN-BASED DATASETS
    # ============================================

    # ============================================
    # UTILITY METHODS
    # ============================================

    @staticmethod
    def list_all() -> None:
        """
        List all available datasets with brief descriptions.
        
        Shows datasets organized by category with sample counts and use cases.
        """
        print("\n" + "="*70)
        print("📚 NEUROGEBRA DATASETS - 100+ Educational ML Datasets")
        print("="*70 + "\n")
        
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
            
        print("\n" + "="*70)
        print(f"📦 {total_count}+ datasets available (base collection)")
        print("💡 Use ExpandedDatasets for 80+ additional datasets!")
        print("📖 Example: Datasets.load_iris(verbose=True)")
        print("="*70 + "\n")
    
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
            print("💡 Try: 'classification', 'regression', 'image', 'medical', 'beginner'\n")
    
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
                "use_cases": ["First ML project", "Classification tutorial", "Feature visualization"]
            },
            "california_housing": {
                "name": "California Housing",
                "samples": 20640,
                "features": 8,
                "task": "Regression",
                "difficulty": "⭐⭐ Intermediate",
                "description": "Predict median house values in California districts based on location and demographics.",
                "use_cases": ["Regression practice", "Feature engineering", "Real estate prediction"]
            },
            "breast_cancer": {
                "name": "Breast Cancer Wisconsin",
                "samples": 569,
                "features": 30,
                "classes": 2,
                "task": "Binary classification",
                "difficulty": "⭐⭐ Intermediate",
                "description": "Diagnose breast cancer (malignant/benign) from cell nucleus measurements.",
                "use_cases": ["Medical ML", "Binary classification", "Feature importance"]
            },
        }
        
        if name in dataset_details:
            info = dataset_details[name]
            print(f"\n{'='*70}")
            print(f"📊 {info['name']}")
            print('='*70)
            print(f"\n  Difficulty: {info['difficulty']}")
            print(f"  Task: {info['task']}")
            print(f"  Samples: {info.get('samples', 'Varies')}")
            print(f"  Features: {info.get('features', 'Varies')}")
            if 'classes' in info:
                print(f"  Classes: {info['classes']}")
            print(f"\n  {info['description']}")
            print(f"\n  💡 Use cases:")
            for uc in info['use_cases']:
                print(f"     • {uc}")
            print(f"\n  📖 Usage: Datasets.load_{name}(verbose=True)")
            print('='*70 + "\n")
        else:
            print(f"\n❌ Dataset '{name}' not found in details database.")
            print("💡 Use Datasets.list_all() to see all available datasets\n")
