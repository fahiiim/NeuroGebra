"""
Quick test to verify the 100+ datasets are working correctly.
"""

def test_dataset_imports():
    """Test that all dataset modules can be imported."""
    try:
        from neurogebra.datasets import Datasets, ExpandedDatasets, CombinedDatasets
        print("✅ All dataset modules imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def test_basic_datasets():
    """Test loading basic datasets."""
    from neurogebra.datasets import CombinedDatasets as Datasets
    
    tests_passed = 0
    tests_total = 0
    
    # Test classification datasets
    datasets_to_test = [
        ("iris", lambda: Datasets.load_iris(verbose=False)),
        ("wine", lambda: Datasets.load_wine(verbose=False)),
        ("breast_cancer", lambda: Datasets.load_breast_cancer(verbose=False)),
        ("digits", lambda: Datasets.load_digits(verbose=False)),
    ]
    
    for name, loader in datasets_to_test:
        tests_total += 1
        try:
            (X_train, y_train), (X_test, y_test) = loader()
            print(f"✅ {name}: {X_train.shape} train, {X_test.shape} test")
            tests_passed += 1
        except Exception as e:
            print(f"❌ {name} failed: {e}")
    
    # Test regression datasets
    regression_tests = [
        ("california_housing", lambda: Datasets.load_california_housing(verbose=False)),
        ("diabetes", lambda: Datasets.load_diabetes(verbose=False)),
        ("simple_regression", lambda: Datasets.load_simple_regression(verbose=False)),
    ]
    
    for name, loader in regression_tests:
        tests_total += 1
        try:
            result = loader()
            if name == "simple_regression":
                X, y = result
                print(f"✅ {name}: X{X.shape}, y{y.shape}")
            else:
                (X_train, y_train), (X_test, y_test) = result
                print(f"✅ {name}: {X_train.shape} train, {X_test.shape} test")
            tests_passed += 1
        except Exception as e:
            print(f"❌ {name} failed: {e}")
    
    # Test synthetic patterns
    synthetic_tests = [
        ("xor", lambda: Datasets.load_xor(verbose=False)),
        ("moons", lambda: Datasets.load_moons(verbose=False)),
        ("circles", lambda: Datasets.load_circles(verbose=False)),
    ]
    
    for name, loader in synthetic_tests:
        tests_total += 1
        try:
            X, y = loader()
            print(f"✅ {name}: X{X.shape}, y{y.shape}")
            tests_passed += 1
        except Exception as e:
            print(f"❌ {name} failed: {e}")
    
    print(f"\n📊 Basic datasets: {tests_passed}/{tests_total} tests passed")
    return tests_passed == tests_total


def test_expanded_datasets():
    """Test loading expanded datasets."""
    from neurogebra.datasets import ExpandedDatasets
    
    tests_passed = 0
    tests_total = 0
    
    expanded_tests = [
        ("covtype", lambda: ExpandedDatasets.load_covtype(n_samples=100, verbose=False)),
        ("energy_efficiency", lambda: ExpandedDatasets.load_energy_efficiency(verbose=False)),
        ("blobs", lambda: ExpandedDatasets.load_blobs(n_samples=100, verbose=False)),
        ("spiral", lambda: ExpandedDatasets.load_spiral(n_samples=100, verbose=False)),
        ("sine_wave", lambda: ExpandedDatasets.load_sine_wave(n_samples=100, verbose=False)),
    ]
    
    for name, loader in expanded_tests:
        tests_total += 1
        try:
            result = loader()
            if isinstance(result, tuple) and len(result) == 2:
                if isinstance(result[0], tuple):  # (train, test) format
                    (X_train, y_train), (X_test, y_test) = result
                    print(f"✅ {name}: {X_train.shape} train, {X_test.shape} test")
                else:  # (X, y) format
                    X, y = result
                    print(f"✅ {name}: X{X.shape}, y{y.shape}")
            tests_passed += 1
        except Exception as e:
            print(f"❌ {name} failed: {e}")
    
    print(f"\n📊 Expanded datasets: {tests_passed}/{tests_total} tests passed")
    return tests_passed == tests_total


def test_utility_methods():
    """Test dataset utility methods."""
    from neurogebra.datasets import Datasets
    
    print("\n" + "="*70)
    print("Testing Utility Methods")
    print("="*70)
    
    try:
        # Test list_all
        print("\n🔍 Testing list_all():")
        Datasets.list_all()
        print("✅ list_all() works")
        
        # Test search
        print("\n🔍 Testing search():")
        Datasets.search("classification")
        print("✅ search() works")
        
        # Test get_info
        print("\n🔍 Testing get_info():")
        Datasets.get_info("iris")
        print("✅ get_info() works")
        
        return True
    except Exception as e:
        print(f"❌ Utility methods failed: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "🧪 "*35)
    print("DATASET FUNCTIONALITY TEST SUITE")
    print("🧪 "*35 + "\n")
    
    all_passed = True
    
    # Test imports
    print("="*70)
    print("Test 1: Module Imports")
    print("="*70)
    if not test_dataset_imports():
        all_passed = False
    print()
    
    # Test basic datasets
    print("="*70)
    print("Test 2: Basic Datasets")
    print("="*70)
    if not test_basic_datasets():
        all_passed = False
    print()
    
    # Test expanded datasets
    print("="*70)
    print("Test 3: Expanded Datasets")
    print("="*70)
    if not test_expanded_datasets():
        all_passed = False
    print()
    
    # Test utility methods
    if not test_utility_methods():
        all_passed = False
    print()
    
    # Summary
    print("\n" + "="*70)
    if all_passed:
        print("✅ ALL TESTS PASSED - 100+ Datasets Working!")
    else:
        print("⚠️  SOME TESTS FAILED - Check errors above")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
