"""
Neurogebra vs NumPy - Comparison Benchmarks
=============================================
Compare Neurogebra's evaluation speed against raw NumPy.
"""

import time

import numpy as np

from neurogebra import MathForge


def benchmark_comparison(neurogebra_fn, numpy_fn, name, x, iterations=100):
    """Compare Neurogebra vs NumPy performance."""
    # Warm up
    for _ in range(5):
        neurogebra_fn(x)
        numpy_fn(x)

    # Neurogebra
    ng_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        neurogebra_fn(x)
        ng_times.append(time.perf_counter() - start)

    # NumPy
    np_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        numpy_fn(x)
        np_times.append(time.perf_counter() - start)

    ng_avg = np.mean(ng_times) * 1000
    np_avg = np.mean(np_times) * 1000
    ratio = ng_avg / np_avg if np_avg > 0 else float("inf")

    print(f"  {name}:")
    print(f"    Neurogebra: {ng_avg:.3f}ms")
    print(f"    NumPy:      {np_avg:.3f}ms")
    print(f"    Ratio:      {ratio:.2f}x")

    return ng_avg, np_avg


def main():
    print("Neurogebra vs NumPy Comparison")
    print("=" * 60)

    forge = MathForge()

    # ReLU
    relu = forge.get("relu")
    x = np.random.randn(100_000)

    print("\nArray size: 100,000 elements")
    print("-" * 40)

    benchmark_comparison(
        lambda x: relu.eval(x=x),
        lambda x: np.maximum(0, x),
        "ReLU",
        x,
    )

    # Sigmoid
    sigmoid = forge.get("sigmoid")
    benchmark_comparison(
        lambda x: sigmoid.eval(x=x),
        lambda x: 1 / (1 + np.exp(-x)),
        "Sigmoid",
        x,
    )

    # Tanh
    tanh_expr = forge.get("tanh")
    benchmark_comparison(
        lambda x: tanh_expr.eval(x=x),
        lambda x: np.tanh(x),
        "Tanh",
        x,
    )

    # Swish
    swish = forge.get("swish")
    benchmark_comparison(
        lambda x: swish.eval(x=x),
        lambda x: x / (1 + np.exp(-x)),
        "Swish",
        x,
    )

    # Softplus
    softplus = forge.get("softplus")
    benchmark_comparison(
        lambda x: softplus.eval(x=x),
        lambda x: np.log(1 + np.exp(x)),
        "Softplus",
        x,
    )

    # Scaling test
    print("\n\nScaling Test (Sigmoid)")
    print("-" * 40)
    sigmoid_expr = forge.get("sigmoid")
    for size in [100, 1_000, 10_000, 100_000, 1_000_000]:
        x = np.random.randn(size)
        ng_t, np_t = benchmark_comparison(
            lambda x: sigmoid_expr.eval(x=x),
            lambda x: 1 / (1 + np.exp(-x)),
            f"n={size:>10,}",
            x,
            iterations=50 if size <= 100_000 else 10,
        )

    print("\n✅ Comparison complete!")


if __name__ == "__main__":
    main()
