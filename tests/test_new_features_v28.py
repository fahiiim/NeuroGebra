"""
Unit tests for NeuroGebra v2.8.0 new features:
- Learning Rate Schedulers (StepLR, CosineAnnealingLR, ExponentialLR, ReduceLROnPlateau)
- Optimizers (SGD, Adam, RMSprop)
- Classification & Regression Metrics
- Advanced Layers (LayerNorm, GlobalAveragePooling2D, ResidualDense)
- Physics-Informed Neural Networks (PINN)
"""

import numpy as np
import pytest

from neurogebra.builders.model_builder import Model, ModelBuilder
from neurogebra.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    explained_variance_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    root_mean_squared_error,
)
from neurogebra.optim import (
    SGD,
    Adam,
    CosineAnnealingLR,
    ExponentialLR,
    ReduceLROnPlateau,
    RMSprop,
    StepLR,
)
from neurogebra.science import PINN


class DummyParam:
    def __init__(self, val):
        self.data = np.array(val, dtype=np.float64)
        self.grad = np.zeros_like(self.data)


def test_lr_schedulers():
    class DummyTarget:
        def __init__(self, lr=0.1):
            self.learning_rate = lr

    # StepLR
    t1 = DummyTarget(0.1)
    sched1 = StepLR(t1, step_size=2, gamma=0.5)
    assert pytest.approx(sched1.current_lr) == 0.1
    assert pytest.approx(sched1.step()) == 0.1
    assert pytest.approx(sched1.step()) == 0.05
    assert pytest.approx(sched1.step()) == 0.05
    assert pytest.approx(sched1.step()) == 0.025

    # CosineAnnealingLR
    t2 = DummyTarget(0.1)
    sched2 = CosineAnnealingLR(t2, T_max=10, eta_min=0.0)
    assert pytest.approx(sched2.current_lr) == 0.1
    # halfway at step 5
    sched2.step(5)
    assert pytest.approx(t2.learning_rate, rel=1e-3) == 0.05
    # at T_max
    sched2.step(10)
    assert pytest.approx(t2.learning_rate, abs=1e-5) == 0.0

    # ExponentialLR
    t3 = DummyTarget(0.1)
    sched3 = ExponentialLR(t3, gamma=0.9)
    sched3.step(1)
    assert pytest.approx(t3.learning_rate) == 0.09
    sched3.step(2)
    assert pytest.approx(t3.learning_rate) == 0.081

    # ReduceLROnPlateau
    t4 = DummyTarget(0.1)
    sched4 = ReduceLROnPlateau(t4, mode="min", factor=0.5, patience=2)
    sched4.step(1.0)
    sched4.step(1.0)
    sched4.step(1.0)
    sched4.step(1.0)  # bad epochs > 2 -> reduction
    assert pytest.approx(t4.learning_rate) == 0.05


def test_optimizers():
    p = DummyParam([1.0, 2.0])
    p.grad = np.array([0.1, -0.2])

    sgd = SGD(lr=0.1, momentum=0.9)
    sgd.step([p])
    assert p.data[0] < 1.0
    assert p.data[1] > 2.0

    p_adam = DummyParam([3.0, 4.0])
    p_adam.grad = np.array([0.5, -0.5])
    adam = Adam(lr=0.01)
    adam.step([p_adam])
    assert p_adam.data[0] < 3.0
    assert p_adam.data[1] > 4.0

    p_rms = DummyParam([2.0, 1.0])
    p_rms.grad = np.array([0.2, -0.2])
    rms = RMSprop(lr=0.01)
    rms.step([p_rms])
    assert p_rms.data[0] < 2.0
    assert p_rms.data[1] > 1.0


def test_classification_metrics():
    y_true = np.array([0, 1, 2, 2, 1, 0])
    y_pred = np.array([0, 1, 1, 2, 1, 0])

    acc = accuracy_score(y_true, y_pred)
    assert pytest.approx(acc) == 5.0 / 6.0

    cm = confusion_matrix(y_true, y_pred, num_classes=3)
    assert cm.shape == (3, 3)
    assert cm[0, 0] == 2
    assert cm[1, 1] == 2
    assert cm[2, 2] == 1
    assert cm[2, 1] == 1

    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    f = f1_score(y_true, y_pred)
    assert 0.0 <= p <= 1.0
    assert 0.0 <= r <= 1.0
    assert 0.0 <= f <= 1.0

    report = classification_report(y_true, y_pred)
    assert "Precision" in report
    assert "Macro Avg" in report


def test_regression_metrics():
    y_true = np.array([3.0, -0.5, 2.0, 7.0])
    y_pred = np.array([2.5, 0.0, 2.0, 8.0])

    mse = mean_squared_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    evs = explained_variance_score(y_true, y_pred)

    assert pytest.approx(rmse) == np.sqrt(mse)
    assert mae > 0
    assert r2 > 0.8
    assert evs > 0.8


def test_advanced_layers():
    builder = ModelBuilder()

    # LayerNorm
    ln = builder.LayerNorm()
    x = np.random.randn(4, 16)
    out_ln = ln.forward(x)
    assert out_ln.shape == (4, 16)
    # output mean near 0 and variance near 1
    assert np.allclose(out_ln.mean(axis=-1), 0.0, atol=1e-4)
    assert np.allclose(out_ln.var(axis=-1), 1.0, atol=1e-3)
    grad_in = ln.backward(np.ones_like(out_ln))
    assert grad_in.shape == x.shape

    # GlobalAveragePooling2D
    gap = builder.GlobalAveragePooling2D()
    x_img = np.random.randn(2, 8, 8, 16)
    out_gap = gap.forward(x_img)
    assert out_gap.shape == (2, 16)
    grad_gap = gap.backward(np.ones_like(out_gap))
    assert grad_gap.shape == x_img.shape

    # ResidualDense
    res_dense = builder.ResidualDense(units=32, activation="relu")
    x_vec = np.random.randn(4, 16)
    out_res = res_dense.forward(x_vec)
    assert out_res.shape == (4, 32)
    grad_res = res_dense.backward(np.ones_like(out_res))
    assert grad_res.shape == x_vec.shape

    # Test in sequential model
    model = Model(name="ResNetTest")
    model.add(builder.Dense(16, activation="relu"))
    model.add(builder.LayerNorm())
    model.add(builder.ResidualDense(16, activation="relu"))
    model.add(builder.Dense(2, activation="softmax"))

    model.compile(loss="categorical_crossentropy", optimizer="adam", learning_rate=0.01)
    X = np.random.randn(20, 8)
    y = np.eye(2)[np.random.randint(0, 2, size=20)]
    sched = StepLR(model, step_size=2, gamma=0.5)
    hist = model.fit(X, y, epochs=3, batch_size=5, lr_scheduler=sched)
    assert len(hist["loss"]) == 3


def test_pinn():
    pinn = PINN(layer_sizes=[1, 16, 1], activation="tanh", seed=42)
    t = np.linspace(0, 1, 20).reshape(-1, 1)
    u = pinn.predict(t)
    assert u.shape == (20, 1)

    # Test gradient and second derivative
    u_t = pinn.compute_gradient(t, var_idx=0)
    u_tt = pinn.compute_second_derivative(t, var_idx=0)
    assert u_t.shape == (20, 1)
    assert u_tt.shape == (20, 1)

    # Test harmonic oscillator residual
    res = pinn.harmonic_oscillator_residual(t, m=1.0, k=4.0, c=0.4)
    assert res.shape == (20, 1)

    # Test train step
    step_res = pinn.train_step(
        X_data=t[:5],
        y_data=np.sin(t[:5]),
        X_colloc=t[5:],
        residual_fn=pinn.harmonic_oscillator_residual,
        lambda_pde=0.1,
        lr=0.01,
    )
    assert "total_loss" in step_res
    assert step_res["total_loss"] >= 0
