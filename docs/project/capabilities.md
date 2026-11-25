# Capabilities and limits

NeuroGebra brings readable mathematics and inspectable training into the same Python workflow. Use this page to choose an execution path before scaling an experiment.

| Subsystem | Available in 3.0 | Practical boundary |
|---|---|---|
| MathForge and Expression | 285 formulas across 10 domains; composition, explanation, LaTeX, symbolic derivatives | Each formula has input and domain assumptions; inspect the formula before using it |
| Value and Tensor | Eager reverse-mode differentiation, broadcasting, reductions, matrix multiplication | NumPy based; rebuild the graph each training step and avoid mutating inputs between forward and backward |
| ModelBuilder and Model | Dense, Conv2D, pooling, flatten, dropout, BatchNorm, LayerNorm, residual dense blocks | Sequential models, NumPy training, NHWC image layout; no distributed training or mixed-precision trainer |
| Optimizers and schedulers | SGD, Adam, RMSprop; step, exponential, cosine, plateau schedules | Optimizer state belongs to an ordered parameter list; compile after the architecture is finalized |
| Metrics | Classification and regression functions using NumPy | Classification accuracy is not a useful regression metric; select metrics appropriate to your task |
| Training Observatory | Training events, formulas, health warnings, summaries and export backends | Detailed logging and full parameter statistics add cost; profile with the logging level you intend to use |
| C JIT | Supported symbolic expressions compiled into native shared libraries | Requires GCC/Clang; uses a portable compiler target rather than forcing AVX2 |
| CUDA JIT | Supported expression kernels compiled through NVRTC | Requires NVIDIA driver, NVRTC and Python bindings; host inputs/results are copied for each evaluation |
| PINN | Boundary/data fitting and ODE/PDE residual minimization | Educational finite-difference solver; coordinate derivatives and residual parameter gradients are numerical, with two residual evaluations per parameter |
| Datasets | 48 loader methods through CombinedDatasets | Many named datasets are synthetic teaching distributions; return conventions vary. See the [catalog](../reference/datasets.md) |
| Bridges | Expression adapters and supported model conversions | Backend support differs; verify predictions, layouts and gradients after conversion |
| Visualization | Matplotlib figures and optional Plotly interactions | Notebook/runtime display support depends on your environment |

## Three independent differentiation paths

`Expression.gradient()` uses SymPy to derive another symbolic expression. `Value` and `Tensor` build eager autograd graphs. `Model` layers implement their own NumPy backward passes. PINN coordinate and residual parameter derivatives use finite differences. These paths share the package but do not form one device-aware execution graph.

## Checkpoint boundaries

`Model.save()` stores architecture, weights, normalization state, history, step count and the configuration/buffers of built-in optimizers. `Model.load()` accepts older checkpoints, which start with fresh optimizer buffers if they were not saved.

Only load trusted checkpoint files: the current format uses Python pickle. Scheduler state, dataset splits, global RNG state and the exact shuffle/dropout sequence are not captured. For reproducible restarts, record them alongside the checkpoint. Custom optimizer checkpointing raises an explicit error.

## Performance claims

No universal speed advantage over PyTorch, TensorFlow or JAX is established by this release. An expression-kernel benchmark measures a different workload from end-to-end model training. Compare the same workload, hardware, precision, transfer boundaries, warmup and correctness tolerance. See the [performance guide](../advanced/performance.md).
