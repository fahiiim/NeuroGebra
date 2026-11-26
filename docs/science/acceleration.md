---
level: Practical guide
runtime: Optional native toolchains
---

# Native expression acceleration

Compile supported symbolic expressions to native C shared libraries or NVIDIA CUDA kernels. These backends apply to expression evaluation; neural-model and PINN training remain NumPy based.

## Inspect the environment

```python
from neurogebra import get_acceleration_info
print(get_acceleration_info())
```

The result includes `c_jit_available`, `c_compiler`, `cuda_available`, `cuda_device` and `cuda_compute_capability`. Treat availability as a starting check; validate the actual expression on the target machine.

## Compile an expression

```python
import numpy as np
from neurogebra import Expression, is_c_available

expr = Expression("energy", "0.5 * m * v**2", params={"m": 2.0})
v = np.linspace(0.0, 10.0, 1000)
baseline = expr.eval(v=v, backend="numpy")

if is_c_available():
    expr.compile(backend="c")
    native = expr.eval(v=v, backend="c")
    np.testing.assert_allclose(native, baseline)
```

## Backend boundaries

| Backend | Precision | Requirements | Current behavior |
|---|---|---|---|
| `numpy` | Usually float64 | Core installation | Vectorized SymPy-generated NumPy callable |
| `c` | float64 | GCC/Clang toolchain, writable cache | Portable native kernel, NumPy broadcasting, disk cache |
| `cuda` | float32 | NVIDIA device/driver, NVRTC, compatible CUDA Python bindings | NVRTC compilation and device launch; host arrays copied for each call |

3.0 uses the compiler's portable CPU target. A compiler accepting `-mavx2` does not establish that a CPU can execute AVX2; the previous forced flag was unsafe on older x86 machines. Compiler identity is now included in the C cache key.

The `fast` extra installs Numba but does not select a Numba backend. MSVC is not detected by the current compiler search, which looks for `clang++`, `g++`, `gcc` and `clang`.

## Low-level functions

```python
import numpy as np
from neurogebra.acceleration import compile_c, is_c_available

if is_c_available():
    fn = compile_c("x*x + y", variables=["x", "y"], opt_level="-O3")
    out = fn(np.array([1.0, 2.0]), 3.0)
    np.testing.assert_allclose(out, [4.0, 7.0])
```

```python
import numpy as np
from neurogebra.acceleration import compile_cuda, is_cuda_available

if is_cuda_available():
    fn = compile_cuda("x*x + 1", variables=["x"])
    x = np.linspace(-1.0, 1.0, 1000, dtype=np.float32)
    out = fn(x)
    np.testing.assert_allclose(out, x*x + 1, rtol=1e-5, atol=1e-6)
```

Supported C optimization arguments are `-O0`, `-O1`, `-O2`, `-O3` and `-Os`. Variables must be unique and cover all free symbols. Unsupported symbolic constructs can fail at code generation or compilation; use the NumPy backend where appropriate.

## Cache and deployment

C artifacts are stored under `~/.neurogebra/cache/c_jit` by default. Set `NEUROGEBRA_CACHE_DIR` to use another root; artifacts use its `c_jit` subdirectory. Treat compiled artifacts as local build products rather than portable wheels. Avoid concurrent processes building the same cold cache entry; atomic multi-process cache coordination is a future improvement.

CUDA context/module lifetime and broader driver-version coverage need further validation. Run native tests on the target hardware before deploying an accelerated workload.

## Benchmark responsibly

Historical kernel speedup figures are workload-specific and are not a validated 3.0 end-to-end training comparison. Use the [performance guide](../advanced/performance.md) to reproduce timings with matched precision, input size, compilation, transfer boundaries and output tolerances.
