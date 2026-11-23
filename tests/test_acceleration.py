"""
Tests for NeuroGebra Acceleration Engine (C-JIT & CUDA-JIT).
"""

import numpy as np
import pytest

from neurogebra.acceleration import (
    compile_c,
    compile_cuda,
    get_acceleration_info,
    is_c_available,
    is_cuda_available,
)
from neurogebra.core.expression import Expression


def test_acceleration_info():
    info = get_acceleration_info()
    assert isinstance(info, dict)
    assert "c_jit_available" in info
    assert "cuda_available" in info


@pytest.mark.skipif(not is_c_available(), reason="C compiler not available")
def test_c_jit_compilation_and_eval():
    c_func = compile_c("x * sin(x) + exp(-x**2)", ["x"])
    x = np.linspace(-3, 3, 50, dtype=np.float64)
    out = c_func(x)
    expected = x * np.sin(x) + np.exp(-(x**2))
    assert np.allclose(out, expected, atol=1e-6)


@pytest.mark.skipif(not is_c_available(), reason="C compiler not available")
def test_c_jit_multi_variable():
    c_func = compile_c("x**2 + y**2 + 2*x*y", ["x", "y"])
    x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    y = np.array([4.0, 5.0, 6.0], dtype=np.float64)
    out = c_func(x, y)
    expected = (x + y) ** 2
    assert np.allclose(out, expected, atol=1e-6)


@pytest.mark.skipif(not is_cuda_available(), reason="CUDA not available")
def test_cuda_jit_compilation_and_eval():
    cu_func = compile_cuda("x / (1.0 + exp(-x))", ["x"])
    x = np.linspace(-3, 3, 50, dtype=np.float32)
    out = cu_func(x)
    expected = x / (1.0 + np.exp(-x))
    assert np.allclose(out, expected, atol=1e-4)


@pytest.mark.skipif(not is_cuda_available(), reason="CUDA not available")
def test_cuda_jit_multi_variable():
    cu_func = compile_cuda("w * x + b", ["w", "x", "b"])
    w = np.float32(2.5)
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.float32(1.0)
    out = cu_func(w, x, b)
    expected = 2.5 * x + 1.0
    assert np.allclose(out, expected, atol=1e-4)


def test_expression_compile_and_eval_backends():
    e = Expression("swish", "x / (1 + exp(-x))")
    x = np.linspace(-2, 2, 20)

    # 1. NumPy backend
    out_np = e.eval(x=x, backend="numpy")
    assert out_np.shape == (20,)

    # 2. C-JIT backend
    if is_c_available():
        e.compile("c")
        out_c = e.eval(x=x, backend="c")
        assert np.allclose(out_np, out_c, atol=1e-5)

    # 3. CUDA-JIT backend
    if is_cuda_available():
        e.compile("cuda")
        out_cu = e.eval(x=x, backend="cuda")
        assert np.allclose(out_np, out_cu, atol=1e-4)
