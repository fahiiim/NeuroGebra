# Framework Bridges

Neurogebra can convert expressions to **PyTorch**, **TensorFlow**, and **JAX** so you can prototype symbolically and integrate with common frameworks.

---

## Why Bridges?

```
Design in Neurogebra (readable, educational)
    ↓
Convert to PyTorch/TF/JAX (fast, production-ready)
    ↓
Deploy in production
```

Support level currently varies by backend, so check the compatibility notes below before production deployment.

---

## Compatibility Notes

- **PyTorch:** Single-input expressions are supported. Gradients flow for input tensor `x` and scalar trainable parameters declared in `Expression(trainable_params=...)`.
- **TensorFlow:** Uses `tf.numpy_function` for interoperability and works in eager/graph execution, but TensorFlow gradients do not propagate through this bridge.
- **JAX:** Uses NumPy-backed eager evaluation for interoperability; not intended for traced `jit`/`grad` workflows.

---

## PyTorch Bridge

### Setup

```bash
pip install neurogebra[frameworks]
# or
pip install torch
```

### Converting Expressions

```python
from neurogebra import MathForge
from neurogebra.bridges.pytorch_bridge import to_pytorch
import torch

forge = MathForge()
sigmoid = forge.get("sigmoid")

# Convert to PyTorch module
torch_sigmoid = to_pytorch(sigmoid)

# Use with PyTorch tensors
x = torch.randn(10)
output = torch_sigmoid(x)
print(output)
```

### Custom Activation as PyTorch Module

```python
from neurogebra import Expression
from neurogebra.bridges.pytorch_bridge import to_pytorch

# Design in Neurogebra
mish = Expression("mish", "x * tanh(log(1 + exp(x)))")

# Convert to PyTorch
mish_module = to_pytorch(mish)

# Use in a PyTorch model
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(784, 128),
    mish_module,               # Your Neurogebra expression!
    nn.Linear(128, 10),
)
```

---

## TensorFlow Bridge

### Setup

```bash
pip install tensorflow
```

### Converting Expressions

```python
from neurogebra import MathForge
from neurogebra.bridges.tensorflow_bridge import to_tensorflow

forge = MathForge()
swish = forge.get("swish")

# Convert to TensorFlow function
tf_swish = to_tensorflow(swish)
```

---

## JAX Bridge

### Setup

```bash
pip install jax jaxlib
```

### Converting Expressions

```python
from neurogebra import MathForge
from neurogebra.bridges.jax_bridge import to_jax

forge = MathForge()
gelu = forge.get("gelu")

# Convert to JAX function
jax_gelu = to_jax(gelu)
```

---

## Workflow: Design → Convert → Deploy

```python
from neurogebra import MathForge, Expression

forge = MathForge()

# 1. DESIGN: Explore and understand
relu = forge.get("relu")
print(relu.explain())
print(relu.symbolic_expr)  # Max(0, x)

# 2. EXPERIMENT: Try different options
for act_name in ["relu", "swish", "gelu"]:
    act = forge.get(act_name)
    print(f"{act_name}(1.0) = {act.eval(x=1.0):.4f}")

# 3. CHOOSE: Pick the best one
chosen = forge.get("swish")

# 4. CONVERT: Move to production framework
from neurogebra.bridges.pytorch_bridge import to_pytorch
torch_activation = to_pytorch(chosen)

# 5. DEPLOY: Use in production model
# model = nn.Sequential(nn.Linear(...), torch_activation, ...)
```

---

## Bridge Comparison

| Feature | PyTorch | TensorFlow | JAX |
|---------|---------|------------|-----|
| Support level | Partial (single-input) | Interop (single-input) | Interop (single-input) |
| Returns | `nn.Module` | TF function | JAX function |
| GPU support | Framework-dependent | Framework-dependent | Framework-dependent |
| Autograd compatible | ✅ (input + scalar trainable params) | ❌ | ❌ |

---

**Next:** [Visualization →](visualization.md)
