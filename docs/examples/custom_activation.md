# Custom Activation Functions

Learn how to create and use custom activation functions with Neurogebra.

## Creating a Custom Activation

```python
from neurogebra import Expression, MathForge

# Define a custom activation
custom_act = Expression(
    name="my_activation",
    symbolic_expr="x * tanh(x)",
    metadata={
        "category": "activation",
        "description": "Custom x*tanh(x) activation function",
        "usage": "Alternative non-monotonic activation",
        "pros": ["Smooth", "Non-monotonic"],
        "cons": ["More expensive than ReLU"],
    }
)

# Use it directly
result = custom_act.eval(x=2.0)
print(f"my_activation(2.0) = {result}")

# Get its gradient
grad = custom_act.gradient("x")
print(f"Gradient formula: {grad.symbolic_expr}")
```

## Registering with MathForge

```python
forge = MathForge()
forge.register("my_activation", custom_act)

# Now it's searchable and accessible
retrieved = forge.get("my_activation")
print(retrieved.explain())
```

## Parametric Activations

```python
# Activation with learnable parameter
parametric = Expression(
    name="parametric_relu",
    symbolic_expr="Max(alpha * x, x)",
    params={"alpha": 0.25},
    trainable_params=["alpha"],
    metadata={
        "category": "activation",
        "description": "Parametric ReLU with learnable slope",
    }
)

# Use with custom alpha
result = parametric.eval(x=-4)  # alpha * (-4) = 0.25 * -4 = -1.0
```

## Comparing Activations

```python
from neurogebra.viz.plotting import plot_comparison

forge = MathForge()

activations = [
    forge.get("relu"),
    forge.get("sigmoid"),
    forge.get("swish"),
    custom_act,
]

fig = plot_comparison(activations, x_range=(-3, 3))
```
