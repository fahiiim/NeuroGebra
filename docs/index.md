---
template: home.html
title: Make the mathematics visible
description: Explore symbolic formulas, inspect gradients, and understand how your model learns with NeuroGebra.
hide:
  - navigation
  - toc
---

<p class="ng-eyebrow">From an equation to an experiment</p>

## Understand it. Run it. Inspect it.

NeuroGebra connects readable mathematics with small, inspectable experiments. Start with a formula, follow its derivatives, build a model, and examine what changed.

<div class="grid cards" markdown="1">

- **01 / Read the formula**

    ---

    Search 285 expressions, compose your own, render LaTeX, and derive exact symbolic gradients with SymPy.

    [Explore symbolic mathematics](tutorial/expressions.md)

- **02 / Follow the gradient**

    ---

    Build scalar and tensor operation graphs. Connect a forward calculation with the reverse-mode derivative that follows it.

    [Understand automatic differentiation](tutorial/autograd.md)

- **03 / Build an experiment**

    ---

    Combine neural layers, choose an optimizer, and train a small NumPy model with predictions and checkpoints you can inspect.

    [Build your first model](tutorial/model-builder.md)

- **04 / Read the training**

    ---

    Bring formulas, gradients, events, and diagnostic summaries into the same workflow with the Training Observatory.

    [See the Observatory](tutorial/training-observatory.md)

- **05 / Explore scientific ideas**

    ---

    Experiment with boundary data and equation residuals using the educational, finite-difference PINN solver.

    [Try a physics-informed network](science/pinn.md)

- **06 / Measure execution**

    ---

    Compile supported symbolic expressions to C or CUDA kernels, and measure performance with explicit precision and hardware boundaries.

    [Choose an execution backend](science/acceleration.md)

</div>

<p class="ng-eyebrow">A path for the way you work</p>

## Choose your next question.

<div class="grid cards" markdown="1">

- **I want to understand the mathematics.**

    Evaluate a formula, inspect its derivative, and connect the result to a small Python program.

    [Take the five-minute tour](getting-started/first-program.md)

- **I want to build and inspect a model.**

    Follow the workflow from array shapes and data preparation through training, evaluation, and observability.

    [Explore the practical guides](tutorials/index.md)

- **I want a complete experiment.**

    Open a focused recipe or a project notebook, then adapt it to the question you are investigating.

    [Browse the example collection](examples/index.md)

</div>

<div class="ng-callout" markdown="1">

**Know your execution path.** Neural models and PINNs train with NumPy. Native C/CUDA acceleration applies to supported expression evaluations. Dataset loaders include generated teaching distributions. Read the [capabilities and limits](project/capabilities.md) before interpreting an experiment.

</div>
