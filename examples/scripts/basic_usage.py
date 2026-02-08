"""
Neurogebra - Basic Usage Example
================================
Demonstrates core features: expressions, evaluation, gradients, and search.
"""

from neurogebra import MathForge, Expression


def main():
    # ── Create MathForge instance ──────────────────────────────────
    forge = MathForge()

    # ── Get pre-built expressions ──────────────────────────────────
    print("=" * 60)
    print("1. Getting Expressions")
    print("=" * 60)

    relu = forge.get("relu")
    sigmoid = forge.get("sigmoid")
    tanh = forge.get("tanh")

    print(f"ReLU:    {relu.symbolic_expr}")
    print(f"Sigmoid: {sigmoid.symbolic_expr}")
    print(f"Tanh:    {tanh.symbolic_expr}")

    # ── Evaluate expressions ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("2. Evaluating Expressions")
    print("=" * 60)

    import numpy as np

    x_vals = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    print(f"x values: {x_vals}")
    print(f"ReLU(x):    {relu.eval(x=x_vals)}")
    print(f"Sigmoid(x): {sigmoid.eval(x=x_vals)}")
    print(f"Tanh(x):    {tanh.eval(x=x_vals)}")

    # ── Compute gradients ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("3. Computing Gradients")
    print("=" * 60)

    sigmoid_grad = sigmoid.gradient("x")
    print(f"d/dx sigmoid(x) = {sigmoid_grad.symbolic_expr}")
    print(f"Gradient at x=0: {sigmoid_grad.eval(x=0)}")

    # ── Compose expressions ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("4. Composing Expressions")
    print("=" * 60)

    mse = forge.get("mse")
    mae = forge.get("mae")
    hybrid = 0.7 * mse + 0.3 * mae
    print(f"Hybrid loss: {hybrid.symbolic_expr}")

    # ── Search ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("5. Searching Expressions")
    print("=" * 60)

    results = forge.search("smooth")
    for name, score in results:
        print(f"  {name}: score={score:.2f}")

    # ── List all ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("6. Listing All Expressions")
    print("=" * 60)

    all_names = forge.list_all()
    print(f"Total expressions: {len(all_names)}")
    print(f"Activations: {forge.list_all(category='activation')}")
    print(f"Losses: {forge.list_all(category='loss')}")

    # ── Custom expression ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("7. Custom Expression")
    print("=" * 60)

    custom = Expression(
        name="my_func",
        symbolic_expr="x**2 + 2*x + 1",
        metadata={"category": "custom", "description": "A simple quadratic"},
    )
    print(f"f(3) = {custom.eval(x=3)}")
    print(f"f'(x) = {custom.gradient('x').symbolic_expr}")
    print(f"Explanation: {custom.explain()}")

    print("\n✅ All basic examples completed successfully!")


if __name__ == "__main__":
    main()
