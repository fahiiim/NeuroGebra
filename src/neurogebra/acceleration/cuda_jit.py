"""
CUDA Acceleration Engine for NeuroGebra.

Compiles SymPy mathematical expressions into high-performance CUDA kernels
and executes them directly on NVIDIA GPUs via NVRTC and CUDA Driver API.
"""

from __future__ import annotations

import ctypes
from typing import Any

import numpy as np
import sympy as sp
from sympy.printing import ccode

_CUDA_INITIALIZED = False
_CUDA_AVAILABLE = False
_CUDA_DEVICE_NAME: str | None = None
_CUDA_COMPUTE_CAP: tuple[int, int] = (0, 0)

try:
    from cuda.bindings import driver, nvrtc

    (res,) = driver.cuInit(0)
    if res.value == 0:
        res, count = driver.cuDeviceGetCount()
        if res.value == 0 and count > 0:
            res, dev = driver.cuDeviceGet(0)
            if res.value == 0:
                res, name = driver.cuDeviceGetName(256, dev)
                dev_name = (
                    name.decode("utf-8").split("\x00")[0].strip()
                    if res.value == 0
                    else "NVIDIA GPU"
                )
                res, major = driver.cuDeviceGetAttribute(
                    driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                    dev,
                )
                res, minor = driver.cuDeviceGetAttribute(
                    driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                    dev,
                )
                _CUDA_AVAILABLE = True
                _CUDA_INITIALIZED = True
                _CUDA_DEVICE_NAME = dev_name
                _CUDA_COMPUTE_CAP = (major, minor)
except Exception:
    _CUDA_AVAILABLE = False


def is_cuda_available() -> bool:
    """Return whether NVIDIA GPU and CUDA acceleration are available."""
    return _CUDA_AVAILABLE


def get_cuda_device_name() -> str | None:
    """Return the name of the active CUDA device."""
    return _CUDA_DEVICE_NAME


def get_cuda_compute_capability() -> tuple[int, int]:
    """Return (major, minor) compute capability of the primary GPU."""
    return _CUDA_COMPUTE_CAP


class CUDAJITFunction:
    """
    Callable wrapper for an NVRTC-compiled CUDA GPU kernel.
    """

    def __init__(
        self,
        ptx_bytes: bytes,
        kernel_name: str,
        var_names: list[str],
    ):
        self.ptx_bytes = ptx_bytes
        self.kernel_name = kernel_name
        self.var_names = var_names
        self.num_vars = len(var_names)

        # Initialize CUDA Context
        assert driver.cuInit(0)[0].value == 0
        err, self.dev = driver.cuDeviceGet(0)
        err, self.ctx = driver.cuCtxCreate(None, 0, self.dev)
        if err.value != 0:
            raise RuntimeError(f"CUDA context creation failed: {err}")

        err, self.module = driver.cuModuleLoadData(self.ptx_bytes)
        if err.value != 0:
            raise RuntimeError(f"CUDA module load failed: {err}")

        err, self.kernel = driver.cuModuleGetFunction(
            self.module, kernel_name.encode("utf-8")
        )
        if err.value != 0:
            raise RuntimeError(f"CUDA get function '{kernel_name}' failed: {err}")

    def __call__(self, *args: Any) -> float | np.ndarray:
        if len(args) != self.num_vars:
            raise ValueError(
                f"Expected {self.num_vars} arguments for variables {self.var_names}, got {len(args)}"
            )

        # Convert args to numpy float32 arrays (standard GPU precision)
        np_args: list[np.ndarray] = []
        max_len = 1
        has_array = False

        for a in args:
            if isinstance(a, np.ndarray):
                arr = np.ascontiguousarray(a, dtype=np.float32)
                if arr.size > 1:
                    has_array = True
                    max_len = max(max_len, arr.size)
                np_args.append(arr)
            elif isinstance(a, (int, float, np.floating, np.integer)):
                arr = np.array([float(a)], dtype=np.float32)
                np_args.append(arr)
            else:
                arr = np.ascontiguousarray(np.array(a, dtype=np.float32))
                if arr.size > 1:
                    has_array = True
                    max_len = max(max_len, arr.size)
                np_args.append(arr)

        # Push context
        driver.cuCtxPushCurrent(self.ctx)
        try:
            d_inputs = []
            for arr in np_args:
                err, d_ptr = driver.cuMemAlloc(arr.nbytes)
                driver.cuMemcpyHtoD(d_ptr, arr.ctypes.data, arr.nbytes)
                d_inputs.append(d_ptr)

            out_bytes = max_len * 4  # float32 = 4 bytes
            err, d_out = driver.cuMemAlloc(out_bytes)

            # Build argument pointers
            # Kernel signature: evaluate_kernel(d_v0, s0, d_v1, s1, ..., d_out, n)
            ctypes_ptrs = []
            c_vars = []

            for i, arr in enumerate(np_args):
                d_val = ctypes.c_uint64(int(d_inputs[i]))
                s_val = ctypes.c_int(1 if arr.size == max_len else 0)
                c_vars.extend([d_val, s_val])
                ctypes_ptrs.extend(
                    [
                        ctypes.cast(ctypes.byref(d_val), ctypes.c_void_p),
                        ctypes.cast(ctypes.byref(s_val), ctypes.c_void_p),
                    ]
                )

            d_out_val = ctypes.c_uint64(int(d_out))
            n_val = ctypes.c_int(max_len)
            c_vars.extend([d_out_val, n_val])
            ctypes_ptrs.extend(
                [
                    ctypes.cast(ctypes.byref(d_out_val), ctypes.c_void_p),
                    ctypes.cast(ctypes.byref(n_val), ctypes.c_void_p),
                ]
            )

            arg_array = (ctypes.c_void_p * len(ctypes_ptrs))(
                *[p.value for p in ctypes_ptrs]
            )

            threads_per_block = 256
            blocks = (max_len + threads_per_block - 1) // threads_per_block

            (launch_err,) = driver.cuLaunchKernel(
                self.kernel,
                blocks,
                1,
                1,
                threads_per_block,
                1,
                1,
                0,
                0,
                ctypes.addressof(arg_array),
                0,
            )
            if launch_err.value != 0:
                raise RuntimeError(f"CUDA kernel launch failed: {launch_err}")

            out = np.empty(max_len, dtype=np.float32)
            driver.cuMemcpyDtoH(out.ctypes.data, d_out, out_bytes)

            for d_ptr in d_inputs:
                driver.cuMemFree(d_ptr)
            driver.cuMemFree(d_out)

        finally:
            driver.cuCtxPopCurrent()

        if not has_array and max_len == 1:
            return float(out[0])

        for a in args:
            if isinstance(a, np.ndarray) and a.size == max_len:
                return out.reshape(a.shape)

        return out


def compile_cuda(
    expr: str | sp.Expr,
    variables: list[str | sp.Symbol],
    arch: str = "compute_80",
) -> CUDAJITFunction:
    """
    Compile a symbolic mathematical expression to an optimized CUDA kernel via NVRTC.

    Args:
        expr: SymPy expression or mathematical string
        variables: List of variable symbols or names
        arch: Target GPU architecture (default compute_80)

    Returns:
        CUDAJITFunction ready to run on the GPU.
    """
    if not is_cuda_available():
        raise RuntimeError(
            "CUDA is not available on this machine. Ensure an NVIDIA GPU and driver are installed."
        )

    if isinstance(expr, str):
        sym_expr = sp.sympify(expr)
    else:
        sym_expr = expr

    var_symbols = [sp.Symbol(v) if isinstance(v, str) else v for v in variables]
    var_names = [v.name for v in var_symbols]

    # Convert SymPy expression to C math code
    c_math = ccode(sym_expr)

    # Build CUDA C source
    # Arguments: const float* v0, int s0, const float* v1, int s1, ..., float* out, int n
    params_sig = []
    for name in var_names:
        params_sig.append(f"const float* _ptr_{name}, int _stride_{name}")
    params_sig.append("float* out, int n")
    sig_str = ", ".join(params_sig)

    cu_lines = [
        f'extern "C" __global__ void evaluate_kernel({sig_str}) {{',
        "    int idx = blockDim.x * blockIdx.x + threadIdx.x;",
        "    if (idx < n) {",
    ]

    for name in var_names:
        cu_lines.append(f"        float {name} = _ptr_{name}[idx * _stride_{name}];")

    cu_lines.append(f"        out[idx] = {c_math};")
    cu_lines.append("    }")
    cu_lines.append("}")
    cu_source = "\n".join(cu_lines)

    # Compile with NVRTC
    err, prog = nvrtc.nvrtcCreateProgram(
        cu_source.encode("utf-8"), b"expr_kernel.cu", 0, [], []
    )
    if err.value != 0:
        raise RuntimeError(f"nvrtcCreateProgram failed: {err}")

    opts = [f"--gpu-architecture={arch}".encode()]
    (comp_err,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
    if comp_err.value != 0:
        err, log_size = nvrtc.nvrtcGetProgramLogSize(prog)
        log = b" " * log_size
        nvrtc.nvrtcGetProgramLog(prog, log)
        raise RuntimeError(f"NVRTC compilation failed:\n{log.decode('utf-8')}")

    err, size = nvrtc.nvrtcGetPTXSize(prog)
    ptx = bytearray(size)
    (err,) = nvrtc.nvrtcGetPTX(prog, ptx)

    return CUDAJITFunction(bytes(ptx), "evaluate_kernel", var_names)
