"""
NeuroGebra Acceleration Subsystem.

Provides ultra-high performance C-JIT (AVX2 SIMD) and CUDA GPU kernel compilation
for symbolic and neural operations.
"""

from __future__ import annotations

from typing import Any

from neurogebra.acceleration.c_jit import (
    CJITFunction,
    compile_c,
    find_c_compiler,
    is_c_available,
)
from neurogebra.acceleration.cuda_jit import (
    CUDAJITFunction,
    compile_cuda,
    get_cuda_compute_capability,
    get_cuda_device_name,
    is_cuda_available,
)


def get_acceleration_info() -> dict[str, Any]:
    """
    Get a summary of available acceleration backends.

    Returns:
        Dictionary detailing C compiler and CUDA GPU capabilities.
    """
    info: dict[str, Any] = {
        "c_jit_available": is_c_available(),
        "c_compiler": find_c_compiler(),
        "cuda_available": is_cuda_available(),
        "cuda_device": get_cuda_device_name(),
        "cuda_compute_capability": get_cuda_compute_capability(),
    }
    return info


__all__ = [
    "CJITFunction",
    "CUDAJITFunction",
    "compile_c",
    "compile_cuda",
    "find_c_compiler",
    "get_acceleration_info",
    "get_cuda_compute_capability",
    "get_cuda_device_name",
    "is_c_available",
    "is_cuda_available",
]
