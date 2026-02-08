"""Tests for statistics repository."""

import pytest
import numpy as np
from neurogebra.repository.statistics import get_statistics_expressions


class TestStatisticsRepository:
    """Test statistical expressions load and evaluate."""

    def setup_method(self):
        self.stats = get_statistics_expressions()

    def test_count_minimum(self):
        assert len(self.stats) >= 30

    def test_all_have_category(self):
        for name, expr in self.stats.items():
            assert "category" in expr.metadata, f"{name} missing category"
            assert expr.metadata["category"] == "statistics"

    def test_normal_pdf_at_mean(self):
        """Standard normal at 0 should be ~0.3989."""
        val = float(self.stats["normal_pdf"].eval(x=0, mu=0, sigma=1))
        assert val == pytest.approx(0.3989, rel=1e-3)

    def test_standard_normal_pdf(self):
        val = float(self.stats["standard_normal_pdf"].eval(x=0))
        assert val == pytest.approx(0.3989, rel=1e-3)

    def test_uniform_pdf(self):
        val = float(self.stats["uniform_pdf"].eval(a=0, b=1))
        assert val == pytest.approx(1.0)

    def test_exponential_pdf_at_zero(self):
        val = float(self.stats["exponential_pdf"].eval(x=0, lambda_param=1))
        assert val == pytest.approx(1.0)

    def test_exponential_cdf(self):
        val = float(self.stats["exponential_cdf"].eval(x=0, lambda_param=1))
        assert val == pytest.approx(0.0, abs=1e-10)

    def test_binary_entropy_max(self):
        """Maximum binary entropy at p=0.5."""
        val = float(self.stats["binary_entropy"].eval(p=0.5))
        assert val == pytest.approx(np.log(2), rel=1e-3)

    def test_z_score(self):
        # Default mu=0, sigma=1 -> z_score(x) = x
        val = float(self.stats["z_score"].eval(x=2))
        assert val == pytest.approx(2.0)

    def test_t_statistic(self):
        # Default mu=0, s=1, n=30 -> t = x * sqrt(30)
        val = float(self.stats["t_statistic"].eval(x=1))
        import numpy as np
        assert val == pytest.approx(np.sqrt(30), rel=1e-3)

    def test_logistic_cdf_at_zero(self):
        val = float(self.stats["logistic_cdf"].eval(x=0, mu=0, s=1))
        assert val == pytest.approx(0.5)

    def test_kl_divergence_zero(self):
        """KL(p||p) = 0."""
        val = float(self.stats["kl_divergence_elem"].eval(p=0.5, q=0.5))
        assert val == pytest.approx(0.0, abs=1e-3)

    def test_linear_regression_pred(self):
        # Default beta_0=0, beta_1=1 -> y=x
        val = float(self.stats["linear_regression_pred"].eval(x=5))
        assert val == pytest.approx(5.0)

    def test_logistic_regression_pred_at_zero(self):
        val = float(self.stats["logistic_regression_pred"].eval(
            x=0, beta_0=0, beta_1=1
        ))
        assert val == pytest.approx(0.5)

    def test_subcategories_exist(self):
        subcats = set()
        for expr in self.stats.values():
            subcats.add(expr.metadata.get("subcategory", ""))
        for expected in ["distribution", "information_theory", "descriptive",
                          "bayesian", "regression"]:
            assert expected in subcats, f"Missing subcategory: {expected}"

    def test_all_expressions_eval(self):
        """Every expression should evaluate without error."""
        for name, expr in self.stats.items():
            syms = [str(s) for s in expr.variables]
            kwargs = {s: 0.5 for s in syms}
            for p, v in expr.params.items():
                kwargs[p] = v
            try:
                result = expr.eval(**kwargs)
                assert np.isfinite(float(result)), f"{name} returned non-finite"
            except Exception as e:
                pytest.fail(f"{name} failed to eval: {e}")
