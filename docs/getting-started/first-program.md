# Your first program

Start with a formula, inspect its derivative, then train a small model. These examples use only the core installation and run on the CPU.

## 1. Read and evaluate a formula

```python
import numpy as np
from neurogebra import MathForge

forge = MathForge()
sigmoid = forge.get("sigmoid")
x = np.array([-2.0, 0.0, 2.0])

print(sigmoid.symbolic_expr)
print(sigmoid.eval(x=x))  # approximately [0.1192, 0.5, 0.8808]
print(sigmoid.latex())
print(sigmoid.explain())
```

The same `Expression` stores a symbolic formula, parameter defaults and explanatory metadata. In Jupyter, displaying the object also renders its LaTeX representation.

## 2. Differentiate symbolically

```python
slope = sigmoid.gradient("x")
print(slope.symbolic_expr)
print(slope.eval(x=0.0))  # 0.25
```

This derivative is another expression. Compute it once and reuse it for array evaluations. Explore the [complete formula catalog](../reference/formulas.md) to find other functions.

## 3. Follow a tensor gradient

```python
from neurogebra.core.autograd import Tensor

x_tensor = Tensor([1.0, 2.0, 3.0], requires_grad=True)
loss = (x_tensor * x_tensor).sum()
loss.backward()
print(x_tensor.grad)  # [2. 4. 6.]
```

Tensor differentiation records operations during the forward pass. It is separate from SymPy differentiation. Leaf gradients accumulate across backward calls; reset them between independent training steps and rebuild the graph after changing parameters.

## 4. Train a small neural model

```python
import numpy as np
from neurogebra import Layer, Model

X = np.linspace(-1.0, 1.0, 64).reshape(-1, 1)
y = 2.0 * X + 1.0

model = Model([Layer("dense", units=1)], name="first_line")
model.compile(loss="mse", optimizer="adam", learning_rate=0.03)
history = model.fit(
    X, y, epochs=40, batch_size=16,
    validation_split=0.2, verbose=False,
)
print(history["loss"][-1])
print(model.predict(np.array([[0.0], [1.0]])))
```

Inputs use `(samples, features)`. Training initializes weights lazily; exact predictions vary with initialization and minibatch order. This example teaches the API rather than guaranteeing a particular final error.

For layer-level events, pass `log_level="basic"` to `compile()`. Detailed Observatory modes collect more statistics and have a higher runtime cost.

## 5. Keep a trusted checkpoint

```python
model.save("first_line.pkl")
restored = Model.load("first_line.pkl")
np.testing.assert_allclose(restored.predict(X), model.predict(X))
```

Only load checkpoint files you trust: the current format uses pickle. Built-in optimizer buffers are preserved in 3.0. Scheduler and RNG state are not part of this checkpoint.

## Continue exploring

| Next question | Guide |
|---|---|
| How do the execution paths fit together? | [Architecture](how-it-works.md) |
| How do I construct a larger model? | [ModelBuilder](../tutorial/model-builder.md) |
| Which data can I load? | [Datasets](../tutorial/datasets.md) |
| How can I see what training does? | [Observatory](../tutorial/training-observatory.md) |
| How do I accelerate an expression? | [Native backends](../science/acceleration.md) |
