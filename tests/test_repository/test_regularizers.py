"""Tests for expanded regularizers repository."""

import pytest
import numpy as np
from neurogebra.repository.regularizers import get_regularizers


class TestRegularizersRepository:
    """Test regularizer expressions load and evaluate."""

    def setup_method(self):
        self.regs = get_regularizers()

    def test_count(self):
        assert len(self.regs) == 20

    def test_all_have_metadata(self):
        for name, expr in self.regs.items():
            assert "category" in expr.metadata, f"{name} missing category"
            assert expr.metadata["category"] == "regularizer"

    def test_l1_sparsity(self):
        l1 = self.regs["l1"]
        assert float(l1.eval(w=0, lambda_reg=0.01)) == 0.0
        assert float(l1.eval(w=5, lambda_reg=0.01)) > 0

    def test_l2_quadratic(self):
        l2 = self.regs["l2"]
        v1 = float(l2.eval(w=1, lambda_reg=0.01))
        v2 = float(l2.eval(w=2, lambda_reg=0.01))
        assert v2 == pytest.approx(4 * v1, rel=1e-6)

    def test_elastic_net_combines(self):
        en = self.regs["elastic_net"]
        result = float(en.eval(w=1, lambda_reg=0.01, alpha_mix=0.5))
        assert result > 0

    def test_weight_decay(self):
        wd = self.regs["weight_decay"]
        assert float(wd.eval(w=0, lambda_reg=0.01)) == 0.0

    def test_entropy_reg(self):
        ent = self.regs["entropy_reg"]
        result = float(ent.eval(p=0.5))
        assert result > 0  # maximum entropy at p=0.5

    def test_group_lasso(self):
        gl = self.regs["group_lasso"]
        # Default lambda_reg=0.01, epsilon=1e-8
        result = float(gl.eval(w1=3, w2=4))
        assert result == pytest.approx(0.01 * 5.0, rel=1e-3)

    def test_max_norm(self):
        mn = self.regs["max_norm"]
        assert float(mn.eval(w=1, max_val=3)) == 0.0  # within limit
        assert float(mn.eval(w=4, max_val=3)) > 0     # exceeds limit

    def test_orthogonal_reg(self):
        oreg = self.regs["orthogonal_reg"]
        # Default lambda_reg=0.01, free vars w1, w2
        result = float(oreg.eval(w1=0.5, w2=1.0))
        assert result > 0

    def test_all_expressions_eval(self):
        """Every regularizer should evaluate without error for basic inputs."""
        for name, expr in self.regs.items():
            syms = [str(s) for s in expr.variables]
            kwargs = {s: 0.5 for s in syms}
            # Override specific params
            for p, v in expr.params.items():
                kwargs[p] = v
            # Ensure w exists
            if "w" in syms:
                kwargs["w"] = 0.5
            try:
                result = expr.eval(**kwargs)
                assert np.isfinite(float(result)), f"{name} returned non-finite"
            except Exception as e:
                pytest.fail(f"{name} failed to eval: {e}")
