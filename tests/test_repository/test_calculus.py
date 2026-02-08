"""Tests for expanded calculus repository."""

import pytest
import numpy as np
from neurogebra.repository.calculus import get_calculus_expressions


class TestCalculusRepository:
    """Test calculus expressions load and evaluate."""

    def setup_method(self):
        self.calc = get_calculus_expressions()

    def test_count_minimum(self):
        assert len(self.calc) >= 45

    def test_all_have_category(self):
        for name, expr in self.calc.items():
            assert "category" in expr.metadata, f"{name} missing category"
            assert expr.metadata["category"] == "calculus"

    def test_exp_at_zero(self):
        assert float(self.calc["exp"].eval(x=0)) == pytest.approx(1.0)

    def test_ln_at_one(self):
        assert float(self.calc["ln"].eval(x=1)) == pytest.approx(0.0)

    def test_sin_at_zero(self):
        assert float(self.calc["sin"].eval(x=0)) == pytest.approx(0.0, abs=1e-10)

    def test_cos_at_zero(self):
        assert float(self.calc["cos"].eval(x=0)) == pytest.approx(1.0)

    def test_sqrt(self):
        assert float(self.calc["sqrt"].eval(x=4)) == pytest.approx(2.0)

    def test_cbrt(self):
        assert float(self.calc["cbrt"].eval(x=8)) == pytest.approx(2.0)

    def test_reciprocal(self):
        assert float(self.calc["reciprocal"].eval(x=4)) == pytest.approx(0.25)

    def test_trig_identity(self):
        """sin²(x) + cos²(x) = 1"""
        x = 0.7
        sin_val = float(self.calc["sin"].eval(x=x))
        cos_val = float(self.calc["cos"].eval(x=x))
        assert sin_val**2 + cos_val**2 == pytest.approx(1.0, rel=1e-10)

    def test_hyperbolic_identity(self):
        """cosh²(x) - sinh²(x) = 1"""
        x = 1.5
        sinh_val = float(self.calc["sinh"].eval(x=x))
        cosh_val = float(self.calc["cosh"].eval(x=x))
        assert cosh_val**2 - sinh_val**2 == pytest.approx(1.0, rel=1e-6)

    def test_erf_bounds(self):
        erf = self.calc["erf"]
        assert float(erf.eval(x=0)) == pytest.approx(0.0, abs=1e-10)
        assert float(erf.eval(x=3)) == pytest.approx(1.0, abs=0.01)

    def test_taylor_exp(self):
        """Taylor series for e^x at x=0 should be close to 1."""
        te = self.calc["taylor_exp"]
        val = float(te.eval(x=0))
        assert val == pytest.approx(1.0, rel=1e-3)

    def test_gradient_descent_step(self):
        gds = self.calc["gradient_descent_step"]
        # Default lr=0.01 -> w - 0.01*gradient
        val = float(gds.eval(w=1.0, gradient=0.5))
        assert val == pytest.approx(0.995)

    def test_finite_difference(self):
        fd = self.calc["finite_difference"]
        # Default h=1e-05 -> (f_plus - f_minus)/(2*h)
        val = float(fd.eval(f_plus=1.00001, f_minus=0.99999))
        assert val == pytest.approx(1.0, rel=1e-2)

    def test_sections_exist(self):
        sections = {
            "elementary": ["exp", "ln", "sqrt", "cbrt"],
            "trig": ["sin", "cos", "tan"],
            "hyperbolic": ["sinh", "cosh", "tanh_func"],
            "special": ["erf", "gamma_func"],
            "taylor": ["taylor_exp", "taylor_sin"],
            "operations": ["gradient_descent_step", "chain_rule"],
        }
        for section, names in sections.items():
            for name in names:
                assert name in self.calc, f"Missing {section}: {name}"
