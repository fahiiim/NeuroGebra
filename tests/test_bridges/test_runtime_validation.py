"""Runtime validation tests for framework bridges."""

import numpy as np
import pytest

from neurogebra.core.expression import Expression


def test_pytorch_bridge_runtime_gradients():
    """Validate PyTorch bridge forward and backward behavior."""
    torch = pytest.importorskip("torch")
    from neurogebra.bridges.pytorch_bridge import to_pytorch

    expr = Expression(
        "affine_quad",
        "a*x**2 + b",
        params={"a": 2.0, "b": 1.0},
        trainable_params=["a", "b"],
    )

    module = to_pytorch(expr)
    x = torch.tensor([1.5, -2.0], dtype=torch.float64, requires_grad=True)
    y = module(x)
    y.sum().backward()

    expected_y = 2.0 * x.detach().numpy() ** 2 + 1.0
    expected_dx = 4.0 * x.detach().numpy()
    expected_da = float((x.detach().numpy() ** 2).sum())
    expected_db = 2.0

    assert np.allclose(y.detach().numpy(), expected_y, atol=1e-6)
    assert np.allclose(x.grad.detach().numpy(), expected_dx, atol=1e-6)
    assert abs(float(module.a.grad) - expected_da) < 1e-5
    assert abs(float(module.b.grad) - expected_db) < 1e-6


def test_tensorflow_bridge_runtime_behavior():
    """Validate TensorFlow bridge eager/graph behavior and gradient semantics."""
    tf = pytest.importorskip("tensorflow")
    from neurogebra.bridges.tensorflow_bridge import to_keras_layer, to_tensorflow

    expr = Expression("swish", "x/(1 + exp(-x))")
    tf_fn = to_tensorflow(expr)

    x = tf.constant([1.0, -2.0], dtype=tf.float32)
    y = tf_fn(x)

    @tf.function
    def traced(v):
        return tf_fn(v)

    y_graph = traced(x)
    expected = expr.eval(x=np.array([1.0, -2.0], dtype=np.float32))

    assert tuple(y.shape) == tuple(x.shape)
    assert tuple(y_graph.shape) == tuple(x.shape)
    assert np.allclose(y.numpy(), expected, atol=1e-6)

    layer = to_keras_layer(expr)
    y_layer = layer(x)
    assert tuple(y_layer.shape) == tuple(x.shape)

    with tf.GradientTape() as tape:
        tape.watch(x)
        loss = tf.reduce_sum(tf_fn(x))
    grad = tape.gradient(loss, x)

    # numpy_function is intentionally non-differentiable in TensorFlow.
    assert grad is None


def test_jax_bridge_runtime_behavior():
    """Validate JAX bridge eager forward and symbolic-gradient outputs."""
    pytest.importorskip("jax")
    import jax.numpy as jnp

    from neurogebra.bridges.jax_bridge import to_jax, to_jax_grad

    expr = Expression("sin_plus", "sin(x) + x**2")
    jax_fn = to_jax(expr)
    jax_grad_fn = to_jax_grad(expr)

    x = jnp.array([0.0, 1.0, -1.0], dtype=jnp.float32)
    y = jax_fn(x)
    grad = jax_grad_fn(x)

    expected_y = np.asarray(
        expr.eval(x=np.array([0.0, 1.0, -1.0], dtype=np.float32))
    )
    expected_grad = np.asarray(
        expr.gradient("x").eval(x=np.array([0.0, 1.0, -1.0], dtype=np.float32))
    )

    assert np.allclose(np.asarray(y), expected_y, atol=1e-6)
    assert np.allclose(np.asarray(grad), expected_grad, atol=1e-6)


def test_bridge_multi_input_guards():
    """All bridges should reject unsupported multi-input expressions."""
    pytest.importorskip("torch")
    pytest.importorskip("tensorflow")
    pytest.importorskip("jax")

    from neurogebra.bridges.jax_bridge import to_jax
    from neurogebra.bridges.pytorch_bridge import to_pytorch
    from neurogebra.bridges.tensorflow_bridge import to_tensorflow

    expr = Expression("multi", "x + y")

    with pytest.raises(ValueError):
        to_pytorch(expr)

    with pytest.raises(ValueError):
        to_tensorflow(expr)

    with pytest.raises(ValueError):
        to_jax(expr)
