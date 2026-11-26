---
level: Practical guide
runtime: NumPy / CPU
---

# Performance engineering

Measure the workload that matters to your users. NeuroGebra 3.0 includes a useful optimization for expression training: analytical derivative expressions and their numerical callables are reused across steps, with current parameter values applied before evaluation.

## Establish a fair baseline

Record hardware, OS, Python/library/compiler versions, array shapes, dtype, batch size, logging level and thread settings. Validate results before comparing time. A float32 CUDA kernel and a float64 NumPy calculation need separate precision notes.

Measure cold compilation and warm execution separately. For GPU work, state whether timing includes host/device transfers and synchronize asynchronous operations when the backend requires it. Measure peak memory and the full training loop as well as a kernel microbenchmark.

## Reproducible expression timing

```python
from time import perf_counter
from statistics import median
import numpy as np
from neurogebra import Expression

expr = Expression("pipeline", "sin(x) * exp(-x*x) + x*x")
x = np.linspace(-3.0, 3.0, 100_000)
expected = np.sin(x) * np.exp(-x*x) + x*x

expr.compile(backend="numpy")
np.testing.assert_allclose(expr.eval(x=x), expected)
times = []
for _ in range(10):
    start = perf_counter()
    result = expr.eval(x=x)
    times.append(perf_counter() - start)
print("NumPy median seconds:", median(times))
```

Repeat with an available native backend, measure its `compile()` separately, use the same input, and check output within an appropriate tolerance. Report the complete setup and raw observations; this documentation does not promise a fixed speedup.

## Reduce avoidable overhead

| Workload | Useful change | Reason |
|---|---|---|
| Formula evaluation | Evaluate a NumPy array in one call | Removes Python iteration and repeated dispatch |
| Repeated derivatives | Create `expr.gradient("x")` once | Avoids repeated symbolic work and compilation |
| Expression parameter training | Reuse one `Trainer` | Its derivative cache survives across steps |
| Model training | Tune batch size on the actual model | Balances matrix utilization, memory and Python overhead |
| Observatory | Use the least detailed level needed | Full array statistics, events and rendering consume time |
| Native kernels | Reuse compiled functions | Amortizes setup costs |
| CUDA expression evaluation | Include transfer cost in decisions | Each call currently moves host inputs/results |
| PINN experiments | Start with small networks and grids | Every residual parameter gradient uses central differences |

## Roadmap toward competitive speed

The next large improvement requires an execution design rather than a language rewrite alone:

1. Profile representative training workloads and publish a versioned benchmark suite with correctness checks.
2. Introduce a backend-neutral tensor/storage interface with explicit device and dtype ownership.
3. Use optimized BLAS/convolution kernels for expensive layer operations; avoid Python loops around tiny operations.
4. Keep tensors and optimizer buffers on the GPU across complete training steps.
5. Add measured fusion for repeated elementwise graphs, followed by gradient correctness tests and hardware CI.

A useful goal is to outperform established frameworks on a clearly defined niche, such as small symbolic expression pipelines with transparent diagnostics. Beating every workload requires substantially broader kernel coverage, memory management, compilation and validation infrastructure.
