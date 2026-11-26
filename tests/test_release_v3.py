"""Regression checks for the correctness changes shipped in 3.0.0."""

import numpy as np
import pytest

from neurogebra import PINN, Layer, Model
from neurogebra.optim import SGD, Adam, RMSprop


@pytest.mark.parametrize(
    "optimizer",
    [
        Adam(lr=0.03, betas=(0.7, 0.8), weight_decay=0.02),
        SGD(lr=0.03, momentum=0.8, nesterov=True),
        RMSprop(lr=0.03, alpha=0.8),
    ],
)
def test_checkpoint_resumes_optimizer_updates(tmp_path, optimizer, capsys):
    model = Model([Layer("dense", units=3), Layer("residual_dense", units=2)])
    model.compile(optimizer=optimizer, metrics=[])
    assert "lr=0.03" in capsys.readouterr().out
    x = np.array([[0.2, 0.4], [0.8, -0.1]])
    y = np.array([[0.0, 1.0], [1.0, 0.0]])
    for _ in range(3):
        model.train_on_batch(x, y)
    path = tmp_path / "model.pkl"
    model.save(path)
    loaded = Model.load(path)
    assert loaded.metrics == []
    assert loaded._optimizer_obj.t == model._optimizer_obj.t
    np.testing.assert_array_equal(model.predict(x), loaded.predict(x))
    assert model.train_on_batch(x, y) == pytest.approx(loaded.train_on_batch(x, y))
    for original, restored in zip(model.get_params(), loaded.get_params()):
        np.testing.assert_array_equal(original.data, restored.data)
    # Snapshot buffers must not alias the live optimizer.
    snapshot = optimizer.state_dict()
    assert snapshot["state"] is not optimizer.__dict__


def test_pinn_large_layer_differentiates_every_weight():
    net = PINN([17, 17, 1])  # 289 weights used to access nonexistent self._rng.
    x = np.ones((2, 17))
    before = net.weights[0].copy()
    net.train_step(x, net.predict(x), x, lambda coords: net.weights[0], lr=0.01)
    assert np.all(net.weights[0] != before)


def test_pinn_multi_output_matches_numerical_gradient():
    net = PINN([2, 3, 2])
    x = np.array([[0.2, -0.3], [0.8, 0.4]])
    y = np.array([[0.0, 1.0], [1.0, 0.0]])
    gradients, _ = net.backward_data(x, y)
    numerical = np.zeros_like(net.weights[0])
    for idx in np.ndindex(numerical.shape):
        original = net.weights[0][idx]
        net.weights[0][idx] = original + 1e-6
        plus = np.mean((net.predict(x) - y) ** 2)
        net.weights[0][idx] = original - 1e-6
        minus = np.mean((net.predict(x) - y) ** 2)
        net.weights[0][idx] = original
        numerical[idx] = (plus - minus) / 2e-6
    np.testing.assert_allclose(gradients[0], numerical, rtol=1e-5, atol=1e-8)


def test_pinn_restores_parameter_and_preserves_callback_error():
    net = PINN([1, 1])
    before = net.weights[0].copy()
    calls = []

    def residual(coords):
        calls.append(1)
        if len(calls) == 2:
            raise TypeError("error inside residual")
        return net.predict(coords)

    with pytest.raises(TypeError, match="error inside residual"):
        net.train_step(np.ones((2, 1)), np.ones((2, 1)), np.ones((2, 1)), residual)
    np.testing.assert_array_equal(net.weights[0], before)
    assert net._step_count == 0
    assert len(calls) == 2


@pytest.mark.parametrize("two_arguments", [False, True])
def test_pinn_residual_signatures_and_vector_targets(two_arguments):
    net = PINN([1, 2, 1])
    x = np.array([[0.0], [1.0]])
    residual = (
        (lambda model, points: model.predict(points)) if two_arguments else net.predict
    )
    result = net.train_step(x, np.ones(2), x, residual)
    assert np.isfinite(result["total_loss"])


@pytest.mark.parametrize("layer_sizes", [[], [1], [1, 0], [1, 1.5]])
def test_invalid_pinn_architecture_rejected(layer_sizes):
    with pytest.raises(ValueError, match="layer_sizes"):
        PINN(layer_sizes)


def test_pinn_rejects_broadcasting_targets():
    net = PINN([1, 2])
    with pytest.raises(ValueError, match="y_data"):
        net.train_step(np.ones((2, 1)), np.ones((2, 1)))


@pytest.mark.parametrize("log_level", [None, "silent"])
def test_softmax_cross_entropy_uses_logits_in_fit_and_evaluate(log_level):
    model = Model([Layer("dense", units=2, activation="softmax")])
    model.compile(
        loss="cross_entropy", optimizer="sgd", learning_rate=0.01, log_level=log_level
    )
    x = np.ones((1, 1))
    model.predict(x)
    model.layers[0].weights[:] = [[-1000.0, 1000.0]]
    model.layers[0].bias[:] = 0.0
    assert model.evaluate(x, np.array([0]))["loss"] == pytest.approx(2000.0)
    history = model.fit(
        x, np.array([0]), epochs=1, batch_size=1, validation_split=0, verbose=False
    )
    assert history["loss"][0] == pytest.approx(2000.0)
    np.testing.assert_allclose(model.layers[0].weights, [[-999.99, 999.99]])


def test_softmax_cross_entropy_soft_targets_have_correct_gradient():
    model = Model([Layer("dense", units=2, activation="softmax")])
    model.compile(loss="cross_entropy", optimizer="sgd", learning_rate=0.1)
    x = np.ones((1, 1))
    model.predict(x)
    model.layers[0].weights[:] = [[-1000.0, 1000.0]]
    model.layers[0].bias[:] = 0.0
    assert model.train_on_batch(x, np.array([[0.25, 0.75]])) == pytest.approx(500.0)
    np.testing.assert_allclose(model.layers[0].weights, [[-999.975, 999.975]])


@pytest.mark.parametrize(
    "targets", [np.array([0.5]), np.array([np.nan]), np.array([0, 1])]
)
def test_cross_entropy_rejects_invalid_labels(targets):
    with pytest.raises(ValueError):
        Model._compute_loss(np.array([[0.2, 0.8]]), targets, "cross_entropy")


def test_softmax_general_vjp_is_not_rewritten_as_cross_entropy():
    layer = Layer("dense", units=2, activation="softmax")
    layer.forward(np.ones((1, 1)))
    layer.weights[:] = [[-40.0, 0.0]]
    p = layer.forward(np.ones((1, 1)))
    upstream = np.array([[-1e6, 0.0]])
    layer.backward(upstream)
    expected = p * (upstream - np.sum(p * upstream, axis=-1, keepdims=True))
    np.testing.assert_allclose(layer.grad_weights, expected, atol=1e-20)


def test_expression_trainer_reuses_derivatives_with_current_parameters(monkeypatch):
    from neurogebra import Expression
    from neurogebra.core.trainer import Trainer

    expr = Expression(
        "coupled", "a*b*x", params={"a": 1.0, "b": 1.0}, trainable_params=["a", "b"]
    )
    gradient = expr.gradient
    calls = []

    def counted(name):
        calls.append(name)
        return gradient(name)

    monkeypatch.setattr(expr, "gradient", counted)
    trainer = Trainer(expr, learning_rate=0.1, optimizer="sgd")
    trainer._backward(np.ones(2), np.zeros(2), "mse")
    trainer._backward(np.ones(2), np.zeros(2), "mse")
    assert calls == ["a", "b"]
    assert expr.params["a"] == pytest.approx(0.6976)
    assert expr.params["b"] == pytest.approx(0.6976)


def test_native_c_compilation_uses_portable_target(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from neurogebra.acceleration import c_jit

    compiler = tmp_path / "compiler"
    compiler.write_text("test compiler", encoding="utf-8")
    monkeypatch.setattr(c_jit, "find_c_compiler", lambda: str(compiler))
    monkeypatch.setattr(c_jit, "_get_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(c_jit, "CJITFunction", lambda *args: args)
    commands = []

    def compile_command(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(c_jit.subprocess, "run", compile_command)
    c_jit.compile_c("x*x", ["x"])
    assert len(commands) == 1
    assert "-mavx2" not in commands[0]
    with pytest.raises(ValueError, match="opt_level"):
        c_jit.compile_c("x", ["x"], opt_level="-march=native")
    with pytest.raises(ValueError, match="unique"):
        c_jit.compile_c("x", ["x", "x"])
    with pytest.raises(ValueError, match="free symbol"):
        c_jit.compile_c("x + y", ["x"])
