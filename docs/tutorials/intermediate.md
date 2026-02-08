# Intermediate Tutorial

This tutorial covers composition, training, and the autograd engine.

## Expression Composition

### Arithmetic Operations

```python
from neurogebra import MathForge, Expression

forge = MathForge()

mse = forge.get("mse")
mae = forge.get("mae")

# Weighted combination
hybrid_loss = 0.7 * mse + 0.3 * mae

# String-based composition
composed = forge.compose("mse + 0.1*mae")
```

### Functional Composition

```python
# f(g(x)) composition
sigmoid = forge.get("sigmoid")
linear = Expression("linear", "2*x + 1")

composed = sigmoid.compose(linear)
# sigmoid(2*x + 1)
result = composed.eval(x=0)
```

## Training Expressions

### Basic Training

```python
import numpy as np
from neurogebra import Expression
from neurogebra.core.trainer import Trainer

# Trainable quadratic
expr = Expression(
    "quadratic",
    "a*x**2 + b*x + c",
    params={"a": 0.0, "b": 0.0, "c": 0.0},
    trainable_params=["a", "b", "c"]
)

# Data from y = x^2 - 2x + 1
X = np.linspace(-3, 3, 50)
y = X**2 - 2*X + 1

# Train with SGD
trainer = Trainer(expr, learning_rate=0.001)
history = trainer.fit(X, y, epochs=500, verbose=True)

print(f"a={expr.params['a']:.2f}, b={expr.params['b']:.2f}, c={expr.params['c']:.2f}")
```

### Using Adam Optimizer

```python
trainer = Trainer(expr, learning_rate=0.01, optimizer="adam")
history = trainer.fit(X, y, epochs=200)
```

## Autograd Engine

### Value Class

```python
from neurogebra.core.autograd import Value

# Create values
x = Value(2.0)
w1 = Value(-3.0)
w2 = Value(1.0)

# Forward pass
h = (w1 * x + w2).relu()
loss = h ** 2

# Backward pass
loss.backward()

print(f"dL/dw1 = {w1.grad}")
print(f"dL/dw2 = {w2.grad}")
print(f"dL/dx = {x.grad}")
```

### Building a Simple Neuron

```python
from neurogebra.core.autograd import Value

# Inputs
x1 = Value(2.0)
x2 = Value(3.0)

# Weights and bias
w1 = Value(0.5)
w2 = Value(-0.3)
b = Value(0.1)

# Neuron computation
z = w1 * x1 + w2 * x2 + b
output = z.sigmoid()

# Backpropagate
output.backward()

print(f"Output: {output.data:.4f}")
print(f"dout/dw1 = {w1.grad:.4f}")
print(f"dout/dw2 = {w2.grad:.4f}")
```

## Custom Expressions

```python
from neurogebra import MathForge, Expression

forge = MathForge()

# Create custom activation
my_act = Expression(
    "parametric_swish",
    "x * (1 / (1 + exp(-beta * x)))",
    params={"beta": 1.0},
    metadata={
        "category": "activation",
        "description": "Parametric Swish with learnable beta",
    }
)

# Register for later use
forge.register("parametric_swish", my_act)

# Now accessible via forge
retrieved = forge.get("parametric_swish")
```

## Next Steps

- [Advanced Tutorial](advanced.md) - Framework bridges, visualization
