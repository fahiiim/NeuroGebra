"""Tests for optimization repository."""

import pytest
import numpy as np
from neurogebra.repository.optimization import get_optimization_expressions


class TestOptimizationRepository:
    """Test optimization expressions load and evaluate."""

    def setup_method(self):
        self.opt = get_optimization_expressions()

    def test_count_minimum(self):
        assert len(self.opt) >= 25

    def test_all_have_category(self):
        for name, expr in self.opt.items():
            assert "category" in expr.metadata, f"{name} missing category"
            assert expr.metadata["category"] == "optimization"

    def test_sgd_step(self):
        # Default lr=0.01 -> w - 0.01*grad
        val = float(self.opt["sgd_step"].eval(w=1.0, grad=0.5))
        assert val == pytest.approx(0.995)

    def test_momentum_step(self):
        # Default lr=0.01, mu_momentum=0.9, v=0 -> w - (0 + 0.01*grad)
        val = float(self.opt["momentum_step"].eval(w=1.0, grad=1.0))
        assert val == pytest.approx(0.99)

    def test_adam_step_identity(self):
        """With zero moments, weight should not change."""
        val = float(self.opt["adam_step"].eval(
            w=1.0, lr=0.001, m_hat=0, v_hat=0
        ))
        assert val == pytest.approx(1.0, abs=1e-3)

    def test_adamw_weight_decay(self):
        val = float(self.opt["adamw_step"].eval(
            w=1.0, lr=0.001, m_hat=0, v_hat=0, lambda_wd=0.01
        ))
        assert val < 1.0  # weight decay shrinks weights

    def test_constant_lr(self):
        # Default lr_init=0.001
        val = float(self.opt["constant_lr"].eval())
        assert val == pytest.approx(0.001)

    def test_exponential_decay_lr(self):
        val0 = float(self.opt["exponential_decay_lr"].eval(
            lr_init=0.01, k_decay=0.01, epoch=0
        ))
        val100 = float(self.opt["exponential_decay_lr"].eval(
            lr_init=0.01, k_decay=0.01, epoch=100
        ))
        assert val0 > val100  # LR decreases over time

    def test_cosine_annealing(self):
        val_start = float(self.opt["cosine_annealing_lr"].eval(
            lr_min=0, lr_max=0.01, T_max=100, epoch=0
        ))
        val_mid = float(self.opt["cosine_annealing_lr"].eval(
            lr_min=0, lr_max=0.01, T_max=100, epoch=50
        ))
        assert val_start == pytest.approx(0.01, rel=1e-3)
        assert val_mid == pytest.approx(0.005, rel=1e-2)

    def test_gradient_clip_value(self):
        clipped = float(self.opt["gradient_clip_value"].eval(
            grad=10, clip_val=1
        ))
        assert clipped == pytest.approx(1.0)

    def test_rosenbrock_minimum(self):
        val = float(self.opt["rosenbrock"].eval(x=1, y=1, a_rb=1, b_rb=100))
        assert val == pytest.approx(0.0, abs=1e-10)

    def test_ackley_minimum(self):
        val = float(self.opt["ackley_2d"].eval(x=0, y=0))
        assert val == pytest.approx(0.0, abs=1e-6)

    def test_subcategories_exist(self):
        subcats = set()
        for expr in self.opt.values():
            subcats.add(expr.metadata.get("subcategory", ""))
        for expected in ["optimizer", "lr_schedule", "gradient_processing", "landscape"]:
            assert expected in subcats, f"Missing subcategory: {expected}"

    def test_all_optimizers_eval(self):
        """All optimizer steps should evaluate."""
        optimizer_names = ["sgd_step", "momentum_step", "nesterov_step",
                           "adagrad_step", "rmsprop_step", "adam_step", "adamw_step"]
        for name in optimizer_names:
            expr = self.opt[name]
            syms = [str(s) for s in expr.variables]
            kwargs = {s: 0.5 for s in syms}
            for p, v in expr.params.items():
                kwargs[p] = v
            kwargs["w"] = 1.0
            kwargs["grad"] = 0.1
            result = float(expr.eval(**kwargs))
            assert np.isfinite(result), f"{name} returned non-finite"
