"""
Neurogebra - Performance Benchmarks
====================================
Benchmarks for expression evaluation, gradient computation, and training.
"""

import time

import numpy as np

from neurogebra import Expression, MathForge
from neurogebra.core.autograd import Value, Tensor
from neurogebra.core.trainer import Trainer


def benchmark(func, name, iterations=100):
    """Run a benchmark and print results."""
    # Warm up
    for _ in range(5):
        func()

    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    avg = np.mean(times) * 1000  # ms
    std = np.std(times) * 1000
    min_t = np.min(times) * 1000
    max_t = np.max(times) * 1000

    print(f"  {name}:")
    print(f"    avg={avg:.3f}ms, std={std:.3f}ms, min={min_t:.3f}ms, max={max_t:.3f}ms")
    return avg


def bench_expression_eval():
    """Benchmark expression evaluation."""
    print("=" * 60)
    print("Benchmark: Expression Evaluation")
    print("=" * 60)

    forge = MathForge()
    sigmoid = forge.get("sigmoid")
    relu = forge.get("relu")
    swish = forge.get("swish")

    x_scalar = 1.0
    x_small = np.random.randn(100)
    x_medium = np.random.randn(10_000)
    x_large = np.random.randn(1_000_000)

    for expr in [relu, sigmoid, swish]:
        print(f"\n  {expr.name}:")
        benchmark(lambda e=expr: e.eval(x=x_scalar), "    scalar", iterations=200)
        benchmark(lambda e=expr: e.eval(x=x_small), "    100 elements", iterations=200)
        benchmark(lambda e=expr: e.eval(x=x_medium), "    10K elements", iterations=100)
        benchmark(lambda e=expr: e.eval(x=x_large), "    1M elements", iterations=20)


def bench_gradient():
    """Benchmark gradient computation."""
    print("\n" + "=" * 60)
    print("Benchmark: Gradient Computation")
    print("=" * 60)

    forge = MathForge()
    sigmoid = forge.get("sigmoid")

    benchmark(lambda: sigmoid.gradient("x"), "  Symbolic gradient (sigmoid)")

    grad = sigmoid.gradient("x")
    x = np.random.randn(10_000)
    benchmark(lambda: grad.eval(x=x), "  Gradient eval (10K elements)")


def bench_autograd():
    """Benchmark autograd engine."""
    print("\n" + "=" * 60)
    print("Benchmark: Autograd Engine")
    print("=" * 60)

    def forward_backward():
        x = Value(2.0)
        w = Value(-3.0)
        b = Value(1.0)
        y = (w * x + b).sigmoid()
        y.backward()

    benchmark(forward_backward, "  Value forward+backward (single neuron)", iterations=500)

    def forward_backward_chain():
        x = Value(1.0)
        for _ in range(10):
            x = (x * Value(0.9) + Value(0.1)).relu()
        x.backward()

    benchmark(forward_backward_chain, "  Value 10-layer chain", iterations=200)

    def tensor_ops():
        a = Tensor(np.random.randn(1000))
        b = Tensor(np.random.randn(1000))
        c = a * b + a
        loss = c.mean()
        loss.backward()

    benchmark(tensor_ops, "  Tensor forward+backward (1K)", iterations=200)


def bench_training():
    """Benchmark trainer."""
    print("\n" + "=" * 60)
    print("Benchmark: Trainer")
    print("=" * 60)

    X = np.linspace(0, 10, 100)
    y = 2 * X + 1

    def train_sgd():
        expr = Expression(
            "line", "m*x + b",
            params={"m": 0.0, "b": 0.0},
            trainable_params=["m", "b"]
        )
        trainer = Trainer(expr, learning_rate=0.001, optimizer="sgd")
        trainer.fit(X, y, epochs=50, verbose=False)

    def train_adam():
        expr = Expression(
            "line", "m*x + b",
            params={"m": 0.0, "b": 0.0},
            trainable_params=["m", "b"]
        )
        trainer = Trainer(expr, learning_rate=0.01, optimizer="adam")
        trainer.fit(X, y, epochs=50, verbose=False)

    benchmark(train_sgd, "  SGD 50 epochs (100 points)", iterations=10)
    benchmark(train_adam, "  Adam 50 epochs (100 points)", iterations=10)


def main():
    print("Neurogebra Performance Benchmarks")
    print("=" * 60)
    print()

    bench_expression_eval()
    bench_gradient()
    bench_autograd()
    bench_training()

    print("\n✅ Benchmarks complete!")


if __name__ == "__main__":
    main()
