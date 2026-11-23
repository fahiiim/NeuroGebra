"""
Example usage of Neurogebra's 100+ datasets.

This script demonstrates how to load and use datasets for
various machine learning tasks.
"""

from neurogebra.datasets import Datasets, ExpandedDatasets, CombinedDatasets
import numpy as np


def example_basic_usage():
    """Basic dataset loading."""
    print("="*70)
    print("EXAMPLE 1: Basic Dataset Loading")
    print("="*70 + "\n")
    
    # Load a simple dataset
    (X_train, y_train), (X_test, y_test) = Datasets.load_iris(verbose=True)
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Number of classes: {len(np.unique(y_train))}\n")


def example_dataset_discovery():
    """Discover available datasets."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Dataset Discovery")
    print("="*70 + "\n")
    
    # List all datasets
    Datasets.list_all()
    
    # Search for specific types
    print("\n Searching for image datasets:")
    Datasets.search("image")
    
    print("\n Searching for medical datasets:")
    Datasets.search("medical")
    
    print("\n Searching for beginner-friendly datasets:")
    Datasets.search("beginner")


def example_classification():
    """Classification dataset examples."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Classification Datasets")
    print("="*70 + "\n")
    
    # Binary classification
    print(" Binary Classification:")
    (X_train, y_train), (X_test, y_test) = Datasets.load_breast_cancer(verbose=True)
    print(f"   Class distribution: {np.bincount(y_train)}\n")
    
    # Multi-class classification
    print(" Multi-class Classification:")
    (X_train, y_train), (X_test, y_test) = Datasets.load_wine(verbose=True)
    print(f"   Classes: {np.unique(y_train)}\n")
    
    # Imbalanced classification
    print(" Imbalanced Classification:")
    (X_train, y_train), (X_test, y_test) = Datasets.load_credit_default(verbose=True)
    print(f"   Class distribution: {np.bincount(y_train)}")
    print(f"   Imbalance ratio: {np.bincount(y_train)[0] / np.bincount(y_train)[1]:.2f}:1\n")


def example_regression():
    """Regression dataset examples."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Regression Datasets")
    print("="*70 + "\n")
    
    # Simple regression
    print(" Simple Regression (for learning):")
    X, y = Datasets.load_simple_regression(n_samples=100, verbose=True)
    print(f"   Feature range: [{X.min():.2f}, {X.max():.2f}]")
    print(f"   Target range: [{y.min():.2f}, {y.max():.2f}]\n")
    
    # Real-world regression
    print(" Real-world Regression:")
    (X_train, y_train), (X_test, y_test) = Datasets.load_california_housing(verbose=True)
    print(f"   Target (house prices) - Mean: ${y_train.mean():.2f}k, Std: ${y_train.std():.2f}k\n")


def example_synthetic_patterns():
    """Synthetic pattern datasets."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Synthetic Pattern Datasets")
    print("="*70 + "\n")
    
    # XOR - non-linear problem
    print(" XOR Pattern:")
    X, y = Datasets.load_xor(n_samples=400, verbose=True)
    print(f"   Perfect for: Learning non-linear decision boundaries\n")
    
    # Moons - curved boundaries
    print(" Moons Pattern:")
    X, y = Datasets.load_moons(n_samples=500, noise=0.15, verbose=True)
    print(f"   Perfect for: Testing non-linear classifiers\n")
    
    # Circles - radial boundaries  
    print(" Circles Pattern:")
    X, y = Datasets.load_circles(n_samples=500, noise=0.05, verbose=True)
    print(f"   Perfect for: Kernel methods, RBF networks\n")


def example_expanded_datasets():
    """Examples using expanded dataset collection."""
    print("\n" + "="*70)
    print("EXAMPLE 6: Extended Dataset Collection")
    print("="*70 + "\n")
    
    # Extended classification datasets
    print(" Extended Classification Datasets:")
    (X_train, y_train), (X_test, y_test) = ExpandedDatasets.load_covtype(n_samples=5000, verbose=True)
    
    # Extended regression datasets
    print("\n Extended Regression Datasets:")
    (X_train, y_train), (X_test, y_test) = ExpandedDatasets.load_energy_efficiency(verbose=True)
    
    # Synthetic patterns
    print("\n Synthetic Pattern Datasets:")
    X, y = ExpandedDatasets.load_spiral(n_samples=1000, verbose=True)
    
    # Time series
    print("\n Time Series Datasets:")
    t, y = ExpandedDatasets.load_sine_wave(n_samples=200, verbose=True)


def example_dataset_info():
    """Get detailed dataset information."""
    print("\n" + "="*70)
    print("EXAMPLE 7: Dataset Information")
    print("="*70 + "\n")
    
    # Get info about specific datasets
    Datasets.get_info("iris")
    Datasets.get_info("california_housing")
    Datasets.get_info("breast_cancer")


def example_combined_interface():
    """Use the combined interface for all datasets."""
    print("\n" + "="*70)
    print("EXAMPLE 8: Combined Dataset Interface")
    print("="*70 + "\n")
    
    print(" CombinedDatasets provides access to ALL 100+ datasets:\n")
    
    # Access base datasets
    (X1, y1), _ = CombinedDatasets.load_iris(verbose=False)
    print(f"[PASS] Loaded iris: {X1.shape}")
    
    # Access expanded datasets
    (X2, y2), _ = CombinedDatasets.load_vehicle(verbose=False)
    print(f"[PASS] Loaded vehicle: {X2.shape}")
    
    # Access synthetic patterns
    X3, y3 = CombinedDatasets.load_spiral(n_samples=500, verbose=False)
    print(f"[PASS] Loaded spiral: {X3.shape}")
    
    # Access time series
    t, y4 = CombinedDatasets.load_sine_wave(n_samples=300, verbose=False)
    print(f"[PASS] Loaded sine_wave: {len(y4)} points")
    
    print("\n Use CombinedDatasets for seamless access to all datasets!")


def main():
    """Run all examples."""
    print("\n" + " "*35)
    print("NEUROGEBRA DATASETS - 100+ Educational Datasets for ML")
    print(" "*35 + "\n")
    
    # Run examples
    example_basic_usage()
    example_dataset_discovery()
    example_classification()
    example_regression()
    example_synthetic_patterns()
    example_expanded_datasets()
    example_dataset_info()
    example_combined_interface()
    
    print("\n" + "="*70)
    print("[PASS] All examples completed!")
    print("="*70)
    print("\n Tips:")
    print("   • Use Datasets.list_all() to browse all datasets")
    print("   • Use Datasets.search('keyword') to find relevant datasets")
    print("   • Use Datasets.get_info('name') for detailed information")
    print("   • Set verbose=False to suppress output messages")
    print("   • All datasets return numpy arrays ready for ML\n")


if __name__ == "__main__":
    main()
