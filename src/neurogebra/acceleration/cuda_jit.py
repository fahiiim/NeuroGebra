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
                # Verify NVRTC is actually working and can compile/destroy a test program
                test_cu = b'extern "C" __global__ void test_k() {}\n'
                err, test_prog = nvrtc.nvrtcCreateProgram(test_cu, b"test.cu", 0, [], [])
                if err.value == 0:
                    nvrtc.nvrtcDestroyProgram(test_prog)
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
        is_all_scalar = True

        for a in args:
            if isinstance(a, np.ndarray):
                arr = np.asanyarray(a, dtype=np.float32)
                if arr.ndim > 0:
                    is_all_scalar = False
                np_args.append(arr)
            elif isinstance(a, (int, float, np.floating, np.integer)):
                np_args.append(np.array(float(a), dtype=np.float32))
            else:
                arr = np.asanyarray(a, dtype=np.float32)
                if arr.ndim > 0:
                    is_all_scalar = False
                np_args.append(arr)

        # Standard NumPy broadcasting
        broadcasted = np.broadcast_arrays(*np_args)
        broadcast_shape = broadcasted[0].shape
        total_len = broadcasted[0].size

        # Handle empty inputs gracefully without GPU kernel launch
        if total_len == 0:
            return np.empty(broadcast_shape, dtype=np.float32)

        b_contiguous = [np.ascontiguousarray(b, dtype=np.float32) for b in broadcasted]

        # Push context
        driver.cuCtxPushCurrent(self.ctx)
        d_inputs = []
        d_out = None
        try:
            for arr in b_contiguous:
                err, d_ptr = driver.cuMemAlloc(arr.nbytes)
                if err.value != 0:
                    raise RuntimeError(f"cuMemAlloc failed: {err}")
                d_inputs.append(d_ptr)
                err = driver.cuMemcpyHtoD(d_ptr, arr.ctypes.data, arr.nbytes)
                if hasattr(err, "value") and err.value != 0:
                    raise RuntimeError(f"cuMemcpyHtoD failed: {err}")

            out_bytes = total_len * 4  # float32 = 4 bytes
            err, d_out = driver.cuMemAlloc(out_bytes)
            if err.value != 0:
                raise RuntimeError(f"cuMemAlloc failed: {err}")

            # Build argument pointers
            ctypes_ptrs = []
            c_vars = []

            for i, _arr in enumerate(b_contiguous):
                d_val = ctypes.c_uint64(int(d_inputs[i]))
                s_val = ctypes.c_int(1)
                c_vars.extend([d_val, s_val])
                ctypes_ptrs.extend(
                    [
                        ctypes.cast(ctypes.byref(d_val), ctypes.c_void_p),
                        ctypes.cast(ctypes.byref(s_val), ctypes.c_void_p),
                    ]
                )

            d_out_val = ctypes.c_uint64(int(d_out))
            n_val = ctypes.c_int(total_len)
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
            blocks = (total_len + threads_per_block - 1) // threads_per_block

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

            out = np.empty(total_len, dtype=np.float32)
            driver.cuMemcpyDtoH(out.ctypes.data, d_out, out_bytes)
        finally:
            for d_ptr in d_inputs:
                try:
                    driver.cuMemFree(d_ptr)
                except Exception:
                    pass
            if d_out is not None:
                try:
                    driver.cuMemFree(d_out)
                except Exception:
                    pass
            driver.cuCtxPopCurrent()

        if is_all_scalar and broadcast_shape == ():
            return float(out[0])

        return out.reshape(broadcast_shape)


def compile_cuda(
    expr: str | sp.Expr,
    variables: list[str | sp.Symbol],
    arch: str | None = None,
) -> CUDAJITFunction:
    """
    Compile a symbolic mathematical expression to an optimized CUDA kernel via NVRTC.

    Args:
        expr: SymPy expression or mathematical string
        variables: List of variable symbols or names
        arch: Target GPU architecture (defaults to active GPU capability or compute_80)

    Returns:
        CUDAJITFunction ready to run on the GPU.
    """
    if not is_cuda_available():
        raise RuntimeError(
            "CUDA is not available on this machine. Ensure an NVIDIA GPU and driver are installed."
        )

    if arch is None:
        major, minor = get_cuda_compute_capability()
        arch = f"compute_{major}{minor}" if major > 0 else "compute_80"

    if isinstance(expr, str):
        sym_expr = sp.sympify(expr)
    else:
        sym_expr = expr

    var_symbols = [sp.Symbol(v) if isinstance(v, str) else v for v in variables]
    var_names = [v.name for v in var_symbols]

    # Mangle symbols to prevent collisions in CUDA kernel
    sym_map = {v: sp.Symbol(f"__ng_v_{k}") for k, v in enumerate(var_symbols)}
    mangled_expr = sym_expr.subs(sym_map)
    c_math = ccode(mangled_expr)

    # Build CUDA C source
    params_sig = []
    for k in range(len(var_names)):
        params_sig.append(f"const float* __ng_ptr_{k}, int __ng_stride_{k}")
    params_sig.append("float* out, int __ng_n")
    sig_str = ", ".join(params_sig)

    cu_lines = [
        f'extern "C" __global__ void evaluate_kernel({sig_str}) {{',
        "    int __ng_idx = blockDim.x * blockIdx.x + threadIdx.x;",
        "    if (__ng_idx < __ng_n) {",
    ]

    for k in range(len(var_names)):
        cu_lines.append(
            f"        float __ng_v_{k} = __ng_ptr_{k}[__ng_idx * __ng_stride_{k}];"
        )

    cu_lines.append(f"        out[__ng_idx] = {c_math};")
    cu_lines.append("    }")
    cu_lines.append("}")
    cu_source = "\n".join(cu_lines)

    # Compile with NVRTC
    err, prog = nvrtc.nvrtcCreateProgram(
        cu_source.encode("utf-8"), b"expr_kernel.cu", 0, [], []
    )
    if err.value != 0:
        raise RuntimeError(f"nvrtcCreateProgram failed: {err}")

    try:
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
    finally:
        nvrtc.nvrtcDestroyProgram(prog)

    return CUDAJITFunction(bytes(ptx), "evaluate_kernel", var_names)
