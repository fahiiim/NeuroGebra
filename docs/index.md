---
title: NeuroGebra — Make the mathematics visible
description: Explore formulas, differentiate tensors, build neural networks, and inspect training with NeuroGebra.
hide:
  - navigation
  - toc
---

<div class="ng-hero" markdown="1">
<div markdown="1">
<div class="ng-eyebrow">NeuroGebra · The 3.0 series</div>

# Make the mathematics visible.

<p class="ng-lead">From a formula you can read to a model you can inspect. Explore symbolic mathematics, build neural networks, and see how learning happens with one open-source Python library.</p>

<div class="ng-hero-actions" markdown="1">

[Start building](getting-started/first-program.md){ .md-button .md-button--primary }
[Explore the API](api/reference.md){ .md-button }

</div>
<p class="ng-small">Python 3.9+ · NumPy + SymPy · MIT licensed</p>
</div>
<div class="ng-equation-card" markdown="1">
<div class="ng-card-label"><span>A formula. An explanation. An executable.</span><span>01 / Symbolic</span></div>
<div class="ng-equation">f(x) = x · σ(x)</div>
<p class="ng-equation-note">Swish: a smooth activation you can evaluate and differentiate.</p>

```python
import numpy as np
from neurogebra import MathForge

swish = MathForge().get("swish")
x = np.linspace(-3.0, 3.0, 7)

values = swish.eval(x=x)
slopes = swish.gradient("x").eval(x=x)
print(swish.latex())
```

</div>
</div>

<div class="ng-facts">
<div><strong>285</strong><span>formulas to explore</span></div>
<div><strong>10</strong><span>mathematical domains</span></div>
<div><strong>48</strong><span>educational dataset loaders</span></div>
<div><strong>20</strong><span>project notebooks</span></div>
</div>

## A working notebook for your ideas

NeuroGebra connects mathematical exploration with small, inspectable experiments. Start with an equation, follow its derivatives, train a model, and examine what changed.

<div class="grid cards" markdown="1">

- **01 · Understand the formula**

    ---

    Search MathForge, compose expressions, render LaTeX, and compute symbolic derivatives with SymPy.

    [Explore expressions](tutorial/expressions.md)

- **02 · Follow the gradient**

    ---

    Work with scalar `Value` objects and multidimensional `Tensor` graphs. Build your intuition for reverse-mode differentiation.

    [Learn autograd](tutorial/autograd.md)

- **03 · Build and train**

    ---

    Combine Dense, Conv2D, normalization, pooling, and residual layers. Use optimizers, schedulers, metrics, and checkpoints.

    [Build a neural network](tutorial/model-builder.md)

- **04 · Inspect the learning**

    ---

    Bring formulas, gradients, health checks, event exports, and training summaries into your workflow with the Observatory.

    [Open the Observatory guide](tutorial/training-observatory.md)

- **05 · Experiment with science**

    ---

    Explore differential-equation residuals with an educational NumPy PINN solver using finite differences.

    [Try a physics-informed model](science/pinn.md)

- **06 · Measure and accelerate**

    ---

    Compile supported symbolic expressions to native C or CUDA kernels. Compare execution paths using reproducible measurements.

    [Understand the backends](science/acceleration.md)

</div>

## Start small. See the result.

=== "Install"

    ```bash
    python -m pip install neurogebra
    python -c "import neurogebra; print(neurogebra.__version__)"
    ```

    For a source checkout, run `python -m pip install -e ".[dev,docs]"`.
    See [installation](getting-started/installation.md) for extras and system requirements.

=== "Differentiate"

    ```python
    from neurogebra.core.autograd import Tensor

    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    loss = (x * x).sum()
    loss.backward()
    print(x.grad)  # [2. 4. 6.]
    ```

=== "Train a model"

    ```python
    import numpy as np
    from neurogebra import Layer, Model

    x = np.linspace(-1.0, 1.0, 64).reshape(-1, 1)
    y = 2.0 * x + 1.0
    model = Model([Layer("dense", units=1)])
    model.compile(loss="mse", optimizer="adam", learning_rate=0.03)
    history = model.fit(x, y, epochs=20, batch_size=16, verbose=False)
    predictions = model.predict(x[:3])
    ```

<div class="ng-callout" markdown="1">

**Know your execution path.** C/CUDA JIT accelerates supported symbolic expression evaluations. `Model.fit()` and `PINN.fit()` use NumPy; installing the CUDA extra does not move their training to a GPU. See [capabilities and limits](project/capabilities.md).

</div>

## Choose your next chapter

| Your goal | A good place to begin |
|---|---|
| Learn the mathematics behind ML | [First program](getting-started/first-program.md), then [beginner track](tutorials/beginner.md) |
| Find a formula or derivative | [MathForge](tutorial/mathforge.md) and [formula catalog](reference/formulas.md) |
| Build an inspectable experiment | [ModelBuilder](tutorial/model-builder.md), [datasets](tutorial/datasets.md), [optimization](advanced/optimization.md) |
| Connect to another framework | [Framework bridges](advanced/framework-bridges.md) |
| Reproduce a complete project | [Notebook collection](examples/project_notebooks.md) |
| Upgrade an existing installation | [3.0.0 release notes](releases/3.0.0.md) and [migration guide](releases/migration-3.md) |
| Contribute a fix or feature | [Contributor guide](project/contributing.md) |

Built by [Fahim Sarker](https://github.com/fahiiim) and the open-source community.
[Source](https://github.com/fahiiim/NeuroGebra) · [PyPI](https://pypi.org/project/neurogebra/) · [Report an issue](https://github.com/fahiiim/NeuroGebra/issues)
