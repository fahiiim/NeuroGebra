"""Tests for linear algebra repository."""

import pytest
import numpy as np
from neurogebra.repository.linalg import get_linalg_expressions


class TestLinalgRepository:
    """Test linear algebra expressions load and evaluate."""

    def setup_method(self):
        self.la = get_linalg_expressions()

    def test_count_minimum(self):
        assert len(self.la) >= 20

    def test_all_have_category(self):
        for name, expr in self.la.items():
            assert "category" in expr.metadata, f"{name} missing category"
            assert expr.metadata["category"] == "linalg"

    def test_dot_product_elem(self):
        val = float(self.la["dot_product_elem"].eval(x=3, y=4))
        assert val == pytest.approx(12.0)

    def test_cosine_similarity_identical(self):
        val = float(self.la["cosine_similarity"].eval(
            dot_xy=1, norm_x=1, norm_y=1
        ))
        assert val == pytest.approx(1.0, rel=1e-3)

    def test_euclidean_dist_same_point(self):
        val = float(self.la["euclidean_dist_elem"].eval(x=5, y=5))
        assert val == pytest.approx(0.0, abs=1e-10)

    def test_manhattan_dist_elem(self):
        val = float(self.la["manhattan_dist_elem"].eval(x=3, y=1))
        assert val == pytest.approx(2.0)

    def test_determinant_2x2(self):
        val = float(self.la["determinant_2x2"].eval(a=1, b=2, c=3, d=4))
        assert val == pytest.approx(-2.0)

    def test_trace_2x2(self):
        val = float(self.la["trace_2x2"].eval(a=3, d=7))
        assert val == pytest.approx(10.0)

    def test_eigenvalue_identity(self):
        """Eigenvalue of identity [[1,0],[0,1]] should be 1."""
        val = float(self.la["eigenvalue_2x2"].eval(a=1, b=0, c=0, d=1))
        assert val == pytest.approx(1.0)

    def test_scaled_dot_product(self):
        # Default dot_qk=1, d_k=64 -> 1/sqrt(64) = 0.125
        val = float(self.la["scaled_dot_product"].eval())
        assert val == pytest.approx(0.125)

    def test_layer_norm_elem(self):
        # Default mu=0, sigma=1, gamma_ln=1, beta_ln=0 -> (x-0)/1 = x
        val = float(self.la["layer_norm_elem"].eval(x=2.0))
        assert val == pytest.approx(2.0, rel=1e-2)

    def test_batch_norm_elem(self):
        # Default mu_batch=0, var_batch=1, gamma_bn=1, beta_bn=0
        # -> (x-0)/sqrt(1+eps) ≈ x
        val = float(self.la["batch_norm_elem"].eval(x=0))
        assert val == pytest.approx(0.0, abs=1e-2)

    def test_softmax_probabilities(self):
        val = float(self.la["softmax_score"].eval(x=0, y=0))
        assert val == pytest.approx(0.5, rel=1e-2)

    def test_subcategories_exist(self):
        subcats = set()
        for expr in self.la.values():
            subcats.add(expr.metadata.get("subcategory", ""))
        for expected in ["norm", "distance", "matrix", "inner_product"]:
            assert expected in subcats, f"Missing subcategory: {expected}"
