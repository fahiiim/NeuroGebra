---
level: Practical guide
runtime: Choose a workflow
---

# Science and speed

Move from an inspectable experiment toward a measured result. These guides cover native expression execution, educational scientific computing, optimization and evaluation.

| Guide | What it covers |
|---|---|
| [Native expression acceleration](acceleration.md) | C/CUDA setup, compilation, precision, cache and hardware boundaries |
| [Physics-informed neural networks](pinn.md) | Data and equation residuals, finite differences, oscillator and Burgers examples |
| [Optimizers and schedules](../advanced/optimization.md) | SGD, Adam, RMSprop and learning-rate policies |
| [Evaluation metrics](metrics.md) | Classification and regression metrics with NumPy |
| [Regularization](../advanced/regularization.md) | Penalties and model complexity |
| [Performance engineering](../advanced/performance.md) | Fair benchmarks, bottlenecks and an execution-backend roadmap |
| [Framework bridges](../advanced/framework-bridges.md) | Supported adapters and conversion limits |
| [Custom expressions](../advanced/custom-expressions.md) | Extend the symbolic building blocks |

Neural models and PINNs train with NumPy. Native acceleration is scoped to supported expression kernels; read the [capability matrix](../project/capabilities.md) before choosing a backend.
