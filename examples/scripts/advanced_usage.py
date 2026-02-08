"""
Neurogebra - Advanced Usage Example
====================================
Demonstrates training, autograd, visualization, and advanced composition.
"""

import numpy as np

from neurogebra import Expression, MathForge
from neurogebra.core.autograd import Value, Tensor
from neurogebra.core.trainer import Trainer


def autograd_demo():
    """Demonstrate the micro autograd engine."""
    print("=" * 60)
    print("1. Autograd Engine - Value")
    print("=" * 60)

    # A simple neuron
    x1 = Value(2.0)
    x2 = Value(3.0)
    w1 = Value(0.5)
    w2 = Value(-0.3)
    b = Value(0.1)

    # Forward
    z = w1 * x1 + w2 * x2 + b
    output = z.sigmoid()

    print(f"z = {z.data:.4f}")
    print(f"sigmoid(z) = {output.data:.4f}")

    # Backward
    output.backward()
    print(f"dout/dw1 = {w1.grad:.4f}")
    print(f"dout/dw2 = {w2.grad:.4f}")
    print(f"dout/db  = {b.grad:.4f}")

    # Tensor operations
    print("\n" + "=" * 60)
    print("2. Autograd Engine - Tensor")
    print("=" * 60)

    a = Tensor(np.array([1.0, 2.0, 3.0]))
    b = Tensor(np.array([4.0, 5.0, 6.0]))
    c = a * b + a
    loss = c.sum()
    loss.backward()

    print(f"a = {a.data}")
    print(f"b = {b.data}")
    print(f"c = a*b + a = {c.data}")
    print(f"loss = sum(c) = {loss.data}")
    print(f"dloss/da = {a.grad}")
    print(f"dloss/db = {b.grad}")


def training_demo():
    """Demonstrate expression training."""
    print("\n" + "=" * 60)
    print("3. Training a Linear Expression")
    print("=" * 60)

    # Create trainable expression
    expr = Expression(
        "fit_line",
        "m*x + b",
        params={"m": 0.0, "b": 0.0},
        trainable_params=["m", "b"],
    )

    # Generate data: y = 2.5x + 1.0
    np.random.seed(42)
    X = np.linspace(0, 10, 100)
    y = 2.5 * X + 1.0 + np.random.normal(0, 0.3, 100)

    # Train with SGD
    trainer = Trainer(expr, learning_rate=0.001, optimizer="sgd")
    history = trainer.fit(X, y, epochs=300, verbose=False)

    print(f"True:    m=2.500, b=1.000")
    print(f"Learned: m={expr.params['m']:.3f}, b={expr.params['b']:.3f}")
    print(f"Final loss: {history['loss'][-1]:.4f}")

    # Train with Adam
    print("\n" + "=" * 60)
    print("4. Training with Adam Optimizer")
    print("=" * 60)

    expr2 = Expression(
        "fit_quad",
        "a*x**2 + b*x + c",
        params={"a": 0.0, "b": 0.0, "c": 0.0},
        trainable_params=["a", "b", "c"],
    )

    X = np.linspace(-3, 3, 100)
    y = 0.5 * X**2 - 2 * X + 1

    trainer2 = Trainer(expr2, learning_rate=0.01, optimizer="adam")
    history2 = trainer2.fit(X, y, epochs=500, verbose=False)

    print(f"True:    a=0.500, b=-2.000, c=1.000")
    print(
        f"Learned: a={expr2.params['a']:.3f}, "
        f"b={expr2.params['b']:.3f}, "
        f"c={expr2.params['c']:.3f}"
    )
    print(f"Final loss: {history2['loss'][-1]:.6f}")


def composition_demo():
    """Demonstrate advanced composition."""
    print("\n" + "=" * 60)
    print("5. Advanced Expression Composition")
    print("=" * 60)

    forge = MathForge()

    # Arithmetic composition
    mse = forge.get("mse")
    mae = forge.get("mae")
    hybrid = 0.8 * mse + 0.2 * mae
    print(f"Hybrid loss: {hybrid.symbolic_expr}")

    # Functional composition
    sigmoid = forge.get("sigmoid")
    linear = Expression("linear_transform", "2*x + 1")
    composed = sigmoid.compose(linear)
    print(f"sigmoid(2x+1) at x=0: {composed.eval(x=0):.4f}")

    # Chained operations
    relu = forge.get("relu")
    scaled = 2 * relu
    shifted = scaled + Expression("offset", "0.5")
    print(f"2*relu(x) + 0.5 at x=1: {shifted.eval(x=1):.4f}")


def search_and_explain_demo():
    """Demonstrate search and explanation."""
    print("\n" + "=" * 60)
    print("6. Search and Explain")
    print("=" * 60)

    forge = MathForge()

    # Search
    results = forge.search("classification")
    print("Search 'classification':")
    for name, score in results[:5]:
        print(f"  {name}: {score:.2f}")

    # Compare
    print("\nComparing activations:")
    comparison = forge.compare(["relu", "sigmoid", "tanh"])
    print(comparison)


def main():
    autograd_demo()
    training_demo()
    composition_demo()
    search_and_explain_demo()
    print("\n✅ All advanced examples completed successfully!")


if __name__ == "__main__":
    main()
