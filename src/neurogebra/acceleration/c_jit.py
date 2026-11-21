"""
C-JIT Acceleration Engine for NeuroGebra.

Compiles SymPy mathematical expressions into optimized native C/C++ shared
libraries using AVX2 SIMD vectorization and OpenMP/Clang loop vectorization.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp
from sympy.printing import ccode

_C_AVAILABLE_CACHE: bool | None = None


def _get_cache_dir() -> Path:
    """Return cache directory for compiled C extensions."""
    cache_dir = Path.home() / ".neurogebra" / "cache" / "c_jit"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def find_c_compiler() -> str | None:
    """Find an available C/C++ compiler on the system."""
    for compiler in ["clang++", "g++", "gcc", "clang"]:
        path = shutil.which(compiler)
        if path:
            return path
    return None


def is_c_available() -> bool:
    """Check if a C/C++ compiler is available and functional for JIT compilation."""
    global _C_AVAILABLE_CACHE
    if _C_AVAILABLE_CACHE is not None:
        return _C_AVAILABLE_CACHE
    compiler = find_c_compiler()
    if compiler is None:
        _C_AVAILABLE_CACHE = False
        return False
    try:
        fn = compile_c("x + 1.0", ["x"])
        res = fn(np.array([1.0, 2.0], dtype=np.float64))
        _C_AVAILABLE_CACHE = bool(np.allclose(res, [2.0, 3.0]))
    except Exception:
        _C_AVAILABLE_CACHE = False
    return _C_AVAILABLE_CACHE


class CJITFunction:
    """
    Callable wrapper for a native C compiled expression.
    """

    def __init__(
        self,
        dll_path: Path,
        func_name: str,
        var_names: list[str],
        dll_handle: ctypes.CDLL | None = None,
    ):
        self.dll_path = dll_path
        self.func_name = func_name
        self.var_names = var_names
        self.num_vars = len(var_names)
        self._dll = dll_handle or ctypes.CDLL(str(dll_path))
        self._c_func = getattr(self._dll, func_name)

        # C signature: void evaluate(const double** in_ptrs, const int* in_strides, double* out, int n)
        self._c_func.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
        ]
        self._c_func.restype = None

    def __call__(self, *args: Any) -> float | np.ndarray:
        if len(args) != self.num_vars:
            raise ValueError(
                f"Expected {self.num_vars} arguments for variables {self.var_names}, got {len(args)}"
            )

        # Convert args to numpy float64 arrays
        np_args: list[np.ndarray] = []
        max_len = 1
        has_array = False

        for a in args:
            if isinstance(a, np.ndarray):
                arr = np.ascontiguousarray(a, dtype=np.float64)
                if arr.size > 1:
                    has_array = True
                    max_len = max(max_len, arr.size)
                np_args.append(arr)
            elif isinstance(a, (int, float, np.floating, np.integer)):
                arr = np.array([float(a)], dtype=np.float64)
                np_args.append(arr)
            else:
                arr = np.ascontiguousarray(np.array(a, dtype=np.float64))
                if arr.size > 1:
                    has_array = True
                    max_len = max(max_len, arr.size)
                np_args.append(arr)

        if not has_array and max_len == 1:
            # All scalar inputs
            in_ptrs_arr = (ctypes.c_void_p * self.num_vars)()
            in_strides_arr = (ctypes.c_int * self.num_vars)()
            for i, arr in enumerate(np_args):
                in_ptrs_arr[i] = arr.ctypes.data
                in_strides_arr[i] = 0

            out_val = np.empty(1, dtype=np.float64)
            self._c_func(
                in_ptrs_arr,
                in_strides_arr,
                out_val.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                1,
            )
            return float(out_val[0])

        # Allocate output buffer
        out = np.empty(max_len, dtype=np.float64)

        in_ptrs_arr = (ctypes.c_void_p * self.num_vars)()
        in_strides_arr = (ctypes.c_int * self.num_vars)()

        for i, arr in enumerate(np_args):
            in_ptrs_arr[i] = arr.ctypes.data
            in_strides_arr[i] = 1 if arr.size == max_len else 0

        self._c_func(
            in_ptrs_arr,
            in_strides_arr,
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            max_len,
        )

        # Match primary array shape if available
        for a in args:
            if isinstance(a, np.ndarray) and a.size == max_len:
                return out.reshape(a.shape)

        return out


def compile_c(
    expr: str | sp.Expr,
    variables: list[str | sp.Symbol],
    opt_level: str = "-O3",
) -> CJITFunction:
    """
    Compile a symbolic expression into a high-performance native C shared library.

    Args:
        expr: SymPy expression or mathematical string
        variables: List of variable symbols or names
        opt_level: Compiler optimization level (default -O3)

    Returns:
        CJITFunction ready for native evaluation.
    """
    compiler = find_c_compiler()
    if compiler is None:
        raise RuntimeError(
            "No C/C++ compiler found (clang++, g++, gcc). Please install a compiler."
        )

    if isinstance(expr, str):
        sym_expr = sp.sympify(expr)
    else:
        sym_expr = expr

    var_symbols = [sp.Symbol(v) if isinstance(v, str) else v for v in variables]
    var_names = [v.name for v in var_symbols]

    # Generate C math body
    c_math = ccode(sym_expr)

    # Build C source
    source_lines = [
        "#include <math.h>",
        "#include <stdlib.h>",
        "",
        "#if defined(_WIN32) || defined(__WIN32__)",
        "#define EXPORT __declspec(dllexport)",
        "#else",
        '#define EXPORT __attribute__((visibility("default")))',
        "#endif",
        "",
        "#ifdef __cplusplus",
        'extern "C" {',
        "#endif",
        "",
        "EXPORT void evaluate_expr(",
        "    const double** in_ptrs,",
        "    const int* in_strides,",
        "    double* out,",
        "    int n",
        ") {",
    ]

    # Declare pointer aliases
    for i, name in enumerate(var_names):
        source_lines.append(f"    const double* _ptr_{name} = in_ptrs[{i}];")
        source_lines.append(f"    int _stride_{name} = in_strides[{i}];")

    source_lines.append("")
    source_lines.append("    #pragma clang loop vectorize(enable)")
    source_lines.append("    #pragma omp simd")
    source_lines.append("    for (int i = 0; i < n; ++i) {")

    for name in var_names:
        source_lines.append(f"        double {name} = _ptr_{name}[i * _stride_{name}];")

    source_lines.append(f"        out[i] = {c_math};")
    source_lines.append("    }")
    source_lines.append("}")
    source_lines.append("")
    source_lines.append("#ifdef __cplusplus")
    source_lines.append("}")
    source_lines.append("#endif")
    source_lines.append("")

    c_source = "\n".join(source_lines)

    # Compute content hash for disk caching
    cache_dir = _get_cache_dir()
    machine = platform.machine().lower()
    src_hash = hashlib.sha256(
        (c_source + opt_level + os.name + machine).encode("utf-8")
    ).hexdigest()[:16]

    if sys.platform == "win32":
        suffix = ".dll"
    elif sys.platform == "darwin":
        suffix = ".dylib"
    else:
        suffix = ".so"

    dll_path = cache_dir / f"neuro_jit_{src_hash}{suffix}"
    src_path = cache_dir / f"neuro_jit_{src_hash}.c"

    if not dll_path.exists():
        src_path.write_text(c_source, encoding="utf-8")

        shared_flag = "-dynamiclib" if sys.platform == "darwin" else "-shared"
        is_x86 = machine in ("x86_64", "amd64", "x64")

        compile_cmd = [compiler, opt_level, shared_flag]
        if sys.platform != "win32":
            compile_cmd.append("-fPIC")
        if is_x86:
            compile_cmd.append("-mavx2")
        compile_cmd.extend([str(src_path), "-o", str(dll_path)])

        proc = subprocess.run(compile_cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"Failed to compile C-JIT kernel: {proc.stderr}\nSource was:\n{c_source}"
            )

    return CJITFunction(dll_path, "evaluate_expr", var_names)
