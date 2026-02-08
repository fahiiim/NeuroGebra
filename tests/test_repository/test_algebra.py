"""Tests for expanded algebra repository."""

import pytest
import numpy as np
from neurogebra.repository.algebra import get_algebra_expressions


class TestAlgebraRepository:
    """Test algebraic expressions load and evaluate."""

    def setup_method(self):
        self.algebra = get_algebra_expressions()

    def test_count_minimum(self):
        assert len(self.algebra) >= 45

    def test_all_have_category(self):
        for name, expr in self.algebra.items():
            assert "category" in expr.metadata, f"{name} missing category"
            assert expr.metadata["category"] == "algebra"

    def test_linear(self):
        lin = self.algebra["linear_eq"]
        # Default params m=1, b=0 -> y=x
        assert float(lin.eval(x=2)) == pytest.approx(2.0)

    def test_quadratic(self):
        quad = self.algebra["quadratic"]
        # Default params a=1, b=0, c=0 -> y=x^2
        assert float(quad.eval(x=3)) == pytest.approx(9.0)

    def test_cubic(self):
        cubic = self.algebra["cubic"]
        # Default params a=1, b=0, c=0, d=0 -> y=x^3
        assert float(cubic.eval(x=2)) == pytest.approx(8.0)

    def test_gaussian(self):
        g = self.algebra["gaussian"]
        val_at_center = float(g.eval(x=0, mu=0, sigma=1, A=1))
        val_away = float(g.eval(x=3, mu=0, sigma=1, A=1))
        assert val_at_center > val_away  # peak at center

    def test_logistic(self):
        log = self.algebra["logistic"]
        val = float(log.eval(x=0, L=1, k=1, x0=0))
        assert val == pytest.approx(0.5, rel=1e-3)

    def test_power_law(self):
        pl = self.algebra["power_law"]
        # Default a_coeff=1, b_exp=2 -> y=x^2
        assert float(pl.eval(x=3)) == pytest.approx(9.0)

    def test_sinusoidal(self):
        sin_expr = self.algebra["sinusoidal"]
        # Default A=1, omega=1, phi=0, offset=0 -> sin(x)
        val = float(sin_expr.eval(x=0))
        assert val == pytest.approx(0.0, abs=1e-10)

    def test_rbf_kernel(self):
        rbf = self.algebra["rbf_kernel"]
        # k(x,x) = 1 for any x (default gamma_param=1)
        val = float(rbf.eval(x=5, y=5))
        assert val == pytest.approx(1.0)

    def test_heaviside(self):
        h = self.algebra["heaviside"]
        # Default threshold=0
        assert float(h.eval(x=1)) == 1.0
        assert float(h.eval(x=-1)) == 0.0

    def test_smoothstep(self):
        ss = self.algebra["smoothstep"]
        val = float(ss.eval(t=0.5))
        assert 0 < val < 1

    def test_lerp(self):
        lerp = self.algebra["lerp"]
        # Default a_val=0, b_val=1 -> lerp(0.5)=0.5
        val = float(lerp.eval(t=0.5))
        assert val == pytest.approx(0.5)

    def test_polynomial_categories_exist(self):
        poly_names = ["linear_eq", "quadratic", "cubic", "quartic", "monomial"]
        for name in poly_names:
            assert name in self.algebra, f"Missing polynomial: {name}"

    def test_distribution_names_exist(self):
        dist_names = ["gaussian", "cauchy_distribution", "student_t",
                       "rayleigh", "laplace_distribution"]
        for name in dist_names:
            assert name in self.algebra, f"Missing distribution: {name}"
