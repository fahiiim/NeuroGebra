"""Tests for metrics repository."""

import pytest
import numpy as np
from neurogebra.repository.metrics import get_metrics_expressions


class TestMetricsRepository:
    """Test metric expressions load and evaluate."""

    def setup_method(self):
        self.met = get_metrics_expressions()

    def test_count_minimum(self):
        assert len(self.met) >= 25

    def test_all_have_category(self):
        for name, expr in self.met.items():
            assert "category" in expr.metadata, f"{name} missing category"
            assert expr.metadata["category"] == "metric"

    def test_precision_perfect(self):
        val = float(self.met["precision_formula"].eval(tp=100, fp=0))
        assert val == pytest.approx(1.0, rel=1e-3)

    def test_recall_perfect(self):
        val = float(self.met["recall_formula"].eval(tp=100, fn=0))
        assert val == pytest.approx(1.0, rel=1e-3)

    def test_f1_perfect(self):
        val = float(self.met["f1_score_formula"].eval(prec=1, rec=1))
        assert val == pytest.approx(1.0, rel=1e-3)

    def test_f1_harmonic_mean(self):
        """F1 with default prec=1, rec=1 -> 2*1*1/(1+1) = 1.0."""
        val = float(self.met["f1_score_formula"].eval())
        assert val == pytest.approx(1.0, rel=1e-2)

    def test_mse_zero_error(self):
        val = float(self.met["mse_elem"].eval(y_pred=5, y_true=5))
        assert val == pytest.approx(0.0, abs=1e-10)

    def test_mae_positive(self):
        val = float(self.met["mae_elem"].eval(y_pred=3, y_true=5))
        assert val == pytest.approx(2.0)

    def test_r_squared_perfect(self):
        val = float(self.met["r_squared"].eval(ss_res=0, ss_tot=100))
        assert val == pytest.approx(1.0, rel=1e-3)

    def test_jaccard_perfect(self):
        val = float(self.met["jaccard_index"].eval(n_intersect=10, n_union=10))
        assert val == pytest.approx(1.0, rel=1e-3)

    def test_dice_coefficient(self):
        # Default n_intersect=1, size_a=1, size_b=1 -> 2*1/(1+1) = 1.0
        val = float(self.met["dice_coefficient"].eval())
        assert val == pytest.approx(1.0, rel=1e-2)

    def test_matthews_perfect(self):
        val = float(self.met["matthews_corr"].eval(tp=50, tn=50, fp=0, fn=0))
        assert val == pytest.approx(1.0, rel=1e-2)

    def test_brier_score_perfect(self):
        val = float(self.met["brier_score_elem"].eval(p=1, y=1))
        assert val == pytest.approx(0.0, abs=1e-10)

    def test_reciprocal_rank(self):
        # Default rank=1 -> 1/1
        val = float(self.met["reciprocal_rank"].eval())
        assert val == pytest.approx(1.0)

    def test_aic(self):
        # Default k=1, log_likelihood=0 -> 2*1 - 2*0 = 2
        val = float(self.met["aic"].eval())
        assert val == pytest.approx(2.0)

    def test_subcategories_exist(self):
        subcats = set()
        for expr in self.met.values():
            subcats.add(expr.metadata.get("subcategory", ""))
        for expected in ["classification", "regression", "similarity", "ranking"]:
            assert expected in subcats, f"Missing subcategory: {expected}"
