# Hardware Acceleration Subsystem

The `neurogebra.acceleration` module compiles supported SymPy expressions into portable native C libraries or NVIDIA CUDA kernels through NVRTC. See the [backend guide](../science/acceleration.md) for precision, toolchain and execution boundaries.

---

## Acceleration Environment Info

::: neurogebra.acceleration.get_acceleration_info
    options:
      show_root_heading: true

---

## C-JIT Engine (Clang / GCC)

::: neurogebra.acceleration.c_jit.compile_c
    options:
      show_root_heading: true

::: neurogebra.acceleration.c_jit.is_c_available
    options:
      show_root_heading: true

::: neurogebra.acceleration.c_jit.find_c_compiler
    options:
      show_root_heading: true

::: neurogebra.acceleration.c_jit.CJITFunction
    options:
      show_root_heading: true
      show_source: true

---

## CUDA-JIT Engine (NVIDIA NVRTC)

::: neurogebra.acceleration.cuda_jit.compile_cuda
    options:
      show_root_heading: true

::: neurogebra.acceleration.cuda_jit.is_cuda_available
    options:
      show_root_heading: true

::: neurogebra.acceleration.cuda_jit.get_cuda_device_name
    options:
      show_root_heading: true

::: neurogebra.acceleration.cuda_jit.get_cuda_compute_capability
    options:
      show_root_heading: true

::: neurogebra.acceleration.cuda_jit.CUDAJITFunction
    options:
      show_root_heading: true
      show_source: true
