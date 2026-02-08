# Advanced Tutorial

This tutorial covers framework bridges, visualization, and advanced patterns.

## Framework Bridges

### PyTorch Integration

```python
from neurogebra import MathForge
from neurogebra.bridges.pytorch_bridge import to_pytorch

forge = MathForge()
sigmoid = forge.get("sigmoid")

# Convert to PyTorch module
torch_sigmoid = to_pytorch(sigmoid)

# Use in PyTorch
import torch
x = torch.randn(10)
output = torch_sigmoid(x)
```

### Creating Custom Layers

```python
from neurogebra import Expression
from neurogebra.bridges.pytorch_bridge import to_pytorch

custom = Expression(
    "custom_act",
    "x * tanh(log(1 + exp(x)))",  # x * tanh(softplus(x)) = Mish
)

# Use as PyTorch module
mish_module = to_pytorch(custom)
```

## Visualization

### Static Plots

```python
from neurogebra import MathForge
from neurogebra.viz.plotting import (
    plot_expression,
    plot_comparison,
    plot_gradient,
    plot_training_history,
)

forge = MathForge()

# Plot single expression
relu = forge.get("relu")
fig = plot_expression(relu, x_range=(-3, 3))

# Compare multiple
activations = [
    forge.get("relu"),
    forge.get("sigmoid"),
    forge.get("tanh"),
    forge.get("swish"),
]
fig = plot_comparison(activations)

# Plot with gradient
sigmoid = forge.get("sigmoid")
fig = plot_gradient(sigmoid)
```

### Interactive Plots

```python
from neurogebra.viz.interactive import interactive_plot, interactive_comparison

# Requires: pip install neurogebra[viz]
fig = interactive_plot(sigmoid, x_range=(-5, 5))
fig.show()
```

## Advanced Composition Patterns

### Building Loss Functions

```python
from neurogebra import MathForge, Expression
from neurogebra.repository.regularizers import get_regularizers

forge = MathForge()

# Base loss
mse = forge.get("mse")

# Add L2 regularization
regs = get_regularizers()
l2 = regs["l2"]

# Custom regularized loss
reg_loss = mse + 0.01 * l2
```

### Expression Calculus

```python
from neurogebra import Expression

# Create expression
f = Expression("gaussian", "exp(-x**2 / 2)")

# Differentiate
f_prime = f.gradient("x")
f_double_prime = f_prime.gradient("x")

# Integrate
F = f.integrate("x")

# Simplify
simplified = f_double_prime.simplify()
```

## Explanation Engine

```python
from neurogebra import MathForge
from neurogebra.utils.explain import ExpressionExplainer

forge = MathForge()
gelu = forge.get("gelu")

# Different formats
text = ExpressionExplainer.explain(gelu, level="advanced", format="text")
md = ExpressionExplainer.explain(gelu, level="advanced", format="markdown")
latex = ExpressionExplainer.explain(gelu, level="advanced", format="latex")

print(md)
```

## Performance Tips

1. **Batch evaluation**: Evaluate with arrays instead of loops
2. **Pre-compile**: Expressions compile on creation; reuse them
3. **Minimize symbolic ops**: Use `.eval()` for numerical work, symbolic only when needed
4. **Use Numba**: Install `neurogebra[fast]` for JIT-compiled evaluation

## Best Practices

- Use `MathForge.search()` to discover expressions
- Always check `.explain()` before using unfamiliar expressions
- Register custom expressions for reuse
- Use the trainer for parameter fitting instead of manual optimization
- Leverage composition for building complex expressions from simple ones
