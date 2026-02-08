"""Tests for transforms repository."""

import pytest
import numpy as np
from neurogebra.repository.transforms import get_transforms_expressions


class TestTransformsRepository:
    """Test transform expressions load and evaluate."""

    def setup_method(self):
        self.tf = get_transforms_expressions()

    def test_count_minimum(self):
        assert len(self.tf) >= 30

    def test_all_have_category(self):
        for name, expr in self.tf.items():
            assert "category" in expr.metadata, f"{name} missing category"
            assert expr.metadata["category"] == "transform"

    def test_min_max_normalize(self):
        # Default x_min=0, x_max=1 -> (x-0)/(1-0) = x
        val = float(self.tf["min_max_normalize"].eval(x=0.5))
        assert val == pytest.approx(0.5, rel=1e-3)

    def test_min_max_bounds(self):
        """Min maps to 0, max maps to 1."""
        # Default x_min=0, x_max=1
        lo = float(self.tf["min_max_normalize"].eval(x=0))
        hi = float(self.tf["min_max_normalize"].eval(x=1))
        assert lo == pytest.approx(0.0, abs=1e-6)
        assert hi == pytest.approx(1.0, rel=1e-3)

    def test_z_score_normalize(self):
        # Default mu=0, sigma=1 -> (x-0)/1 = x
        val = float(self.tf["z_score_normalize"].eval(x=2.0))
        assert val == pytest.approx(2.0, rel=1e-3)

    def test_robust_scale(self):
        # Default median_val=0, iqr=1 -> (x-0)/1 = x
        val = float(self.tf["robust_scale"].eval(x=2.0))
        assert val == pytest.approx(2.0, rel=1e-3)

    def test_log_transform_at_zero(self):
        val = float(self.tf["log_transform"].eval(x=0))
        assert val == pytest.approx(0.0, abs=1e-10)

    def test_sigmoid_transform_at_zero(self):
        val = float(self.tf["sigmoid_transform"].eval(x=0, k_sig=1, x0_sig=0))
        assert val == pytest.approx(0.5)

    def test_tanh_transform_bounds(self):
        val = float(self.tf["tanh_transform"].eval(x=100))
        assert val == pytest.approx(1.0, rel=1e-3)

    def test_clip_transform(self):
        val = float(self.tf["clip_transform"].eval(x=10, low=0, high=1))
        assert val == pytest.approx(1.0)
        val2 = float(self.tf["clip_transform"].eval(x=-5, low=0, high=1))
        assert val2 == pytest.approx(0.0)

    def test_xavier_normal_std(self):
        val = float(self.tf["xavier_normal_std"].eval(fan_in=256, fan_out=256))
        expected = np.sqrt(2 / 512)
        assert val == pytest.approx(expected, rel=1e-3)

    def test_he_normal_std(self):
        val = float(self.tf["he_normal_std"].eval(fan_in=256))
        expected = np.sqrt(2 / 256)
        assert val == pytest.approx(expected, rel=1e-3)

    def test_exponential_smoothing(self):
        # Default alpha_es=0.3, s_prev=0 -> 0.3*x + 0.7*0 = 0.3*x
        val = float(self.tf["exponential_smoothing"].eval(x=10))
        assert val == pytest.approx(3.0)

    def test_log_return(self):
        # Default x_prev=1 -> log(x/1) = log(x)
        val = float(self.tf["log_return"].eval(x=np.e))
        assert val == pytest.approx(1.0, rel=1e-3)

    def test_gaussian_basis_at_center(self):
        val = float(self.tf["gaussian_basis"].eval(x=0, center=0, width=1))
        assert val == pytest.approx(1.0)

    def test_subcategories_exist(self):
        subcats = set()
        for expr in self.tf.values():
            subcats.add(expr.metadata.get("subcategory", ""))
        for expected in ["normalization", "power", "nonlinear", "encoding",
                          "initialization"]:
            assert expected in subcats, f"Missing subcategory: {expected}"

    def test_all_expressions_eval(self):
        """Every transform should evaluate without error."""
        for name, expr in self.tf.items():
            syms = [str(s) for s in expr.variables]
            kwargs = {s: 0.5 for s in syms}
            for p, v in expr.params.items():
                kwargs[p] = v
            try:
                result = expr.eval(**kwargs)
                assert np.isfinite(float(result)), f"{name} returned non-finite"
            except Exception as e:
                pytest.fail(f"{name} failed to eval: {e}")
