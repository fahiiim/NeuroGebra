# Hardware Acceleration Engine (C-JIT & CUDA GPU)

NeuroGebra bridges symbolic interpretability and bare-metal execution speed through `neurogebra.acceleration`—a zero-configuration native compiler that transforms SymPy expressions into optimized C++ shared libraries (AVX2 SIMD / OpenMP) or NVIDIA CUDA device kernels via NVRTC.

---

## Performance Summary

Evaluated on 1,000,000 float elements computing composite non-linear activation pipelines (Intel i7 + NVIDIA GeForce RTX 5070 12GB):

| Execution Backend | Compilation Pipeline | Latency (1M elements) | Relative Speedup |
|---|---|---:|---:|
| **NumPy (Baseline)** | Compiled Python `lambdify` + ufuncs | 10.55 ms | 1.00x |
| **NeuroGebra C-JIT (`backend="c"`)** | Clang++ / GCC `-O3` AVX2 SIMD + OpenMP | 5.16 ms | **2.04x faster** |
| **NeuroGebra CUDA-JIT (`backend="cuda"`)** | Fused PTX kernel via NVRTC + Driver API | 2.22 ms | **4.76x faster** |

---

## Inspecting Local Hardware Capabilities

Check whether a native C/C++ compiler (`clang++`, `g++`, `gcc`, `clang`) or NVIDIA CUDA Driver + NVRTC stack is detected on your machine:

```python
from neurogebra import (
    get_acceleration_info,
    is_c_available,
    is_cuda_available,
)

info = get_acceleration_info()
print(info)
# {
#     'c_jit_available': True,
#     'c_compiler': '/usr/bin/g++',
#     'cuda_available': True,
#     'cuda_device': 'NVIDIA GeForce RTX 5070',
#     'cuda_compute_capability': (8, 9),
# }
```

---

## Accelerating Symbolic `Expression` Objects

Any symbolic `Expression` can compile directly onto C or CUDA using `.compile(backend=...)` or per-call `backend=...`:

```python
import numpy as np
from neurogebra import Expression

# Define symbolic expression (supports multi-variable formulas and math functions)
swish = Expression("swish", "x / (1.0 + exp(-beta * x))", params={"beta": 1.0})
data = np.linspace(-5.0, 5.0, 1_000_000)

# Option 1: Compile and cache C-JIT shared library
swish.compile(backend="c")
cpu_out = swish.eval(x=data)

# Option 2: Compile PTX GPU kernel via NVRTC and launch on CUDA device
swish.compile(backend="cuda")
gpu_out = swish.eval(x=data)

# Option 3: Override backend per evaluation call
np_out = swish.eval(x=data, backend="numpy")
```

### Supported Compilation Backends

| Backend | Precision | Broadcasting | Caching Behavior |
|---|---|---|---|
| `"numpy"` | `float64` | Full NumPy broadcasting | In-memory `lambdify` callable |
| `"c"` | `float64` | Full `np.broadcast_arrays` | Disk-cached `.so` / `.dll` / `.dylib` under `~/.neurogebra/cache/c_jit` keyed by SHA-256 of source & ISA flags |
| `"cuda"` | `float32` | Full `np.broadcast_arrays` | NVRTC PTX compiled to CUDA module with automatic `cuMemAlloc` / `cuMemFree` cleanup |

---

## Direct Low-Level JIT Functions (`compile_c` & `compile_cuda`)

For raw string expressions or SymPy graphs outside `Expression`, use the direct compiler functions:

### C-JIT Compiler

```python
import numpy as np
from neurogebra.acceleration import compile_c

# Compile a multi-variable formula with symbol mangling safety (variables like 'i' or 'n' are safe)
energy_fn = compile_c("0.5 * m * v**2", variables=["m", "v"], opt_level="-O3")

masses = np.array([1.0, 2.5, 4.0])
velocities = np.linspace(0.0, 10.0, 3)
print(energy_fn(masses, velocities))
```

### CUDA NVRTC Kernel Compiler

```python
import numpy as np
from neurogebra.acceleration import compile_cuda, is_cuda_available

if is_cuda_available():
    kernel_fn = compile_cuda("exp(-x**2 - y**2) * cos(x * y)", variables=["x", "y"])
    x = np.linspace(-2.0, 2.0, 100_000, dtype=np.float32)
    y = np.linspace(-2.0, 2.0, 100_000, dtype=np.float32)
    out = kernel_fn(x, y)
```

---

## Installation & System Prerequisites

- **C-JIT CPU**: Any standard host toolchain (`gcc`, `g++`, `clang`, or `clang++`) visible on `PATH`. On x86-64 hosts, NeuroGebra compiles with `-O3 -mavx2` and automatically falls back to baseline CPU optimization flags if an instruction set is unavailable.
- **CUDA-JIT GPU**: Install CUDA Python bindings and NVIDIA driver runtime (`pip install "neurogebra[cuda]"`). Target architecture defaults automatically to your active GPU's compute capability.
