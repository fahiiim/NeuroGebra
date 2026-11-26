---
level: Example collection
runtime: Choose a workflow below
description: Small NeuroGebra recipes and complete project notebooks, organized by what you want to try.
---

# A starting point for your experiment

Choose a small recipe to learn one idea, or a project notebook to follow a complete workflow. Start with the simplest example that answers your question.

## Focused recipes

<div class="grid cards" markdown="1">

- **Create an activation**

    Define a formula, inspect its behavior, and connect it to a derivative.

    [Custom activation functions](custom_activation.md)

- **Design a loss**

    Express the objective you want to minimize and understand its assumptions.

    [Custom loss functions](custom_loss.md)

- **Fit a formula**

    Turn symbolic parameters into a small trainable experiment.

    [Training expressions](training_expressions.md)

</div>

## Complete projects

| Start here | What you will explore |
|---|---|
| [Linear regression](../projects/project1-linear-regression.md) | Data, parameters, optimization and a comparison workflow |
| [Image classification](../projects/project2-image-classifier.md) | Image shapes, layers and evaluation |
| [A neural network](../projects/project3-neural-network.md) | A model's forward, backward and update steps |
| [20 project notebooks](project_notebooks.md) | Symbolic calculus, autograd, training, science and framework integrations |

The notebooks have different optional dependencies and execution paths. Check their setup cells and the [capability matrix](../project/capabilities.md). A notebook using another framework's GPU runtime does not imply that NeuroGebra's NumPy model training runs on the GPU.

## Inspect an example in your browser

The [homepage activation explorer](../index.md) displays library-generated function and derivative samples. It is a lightweight illustration, with the equivalent Python shown alongside it. For a live Python experiment, use a local environment or one of the notebook links.

## Before comparing results

Record the package version, backend, seed, dataset provenance, dtype and preprocessing. Use the [performance guide](../advanced/performance.md) for timing comparisons and the [troubleshooting guide](../project/troubleshooting.md) when an example behaves differently in your environment.
