"""Tests for loss function repository."""

import pytest
import numpy as np
from neurogebra.repository.losses import get_losses


class TestLossRepository:
    """Test loss function repository."""

    def test_get_losses_returns_dict(self):
        """Test that get_losses returns a dictionary."""
        loss_fns = get_losses()
        assert isinstance(loss_fns, dict)
        assert len(loss_fns) > 0

    def test_all_are_expressions(self):
        """Test all values are Expression instances."""
        from neurogebra.core.expression import Expression

        loss_fns = get_losses()
        for name, expr in loss_fns.items():
            assert isinstance(expr, Expression), f"{name} is not an Expression"

    def test_all_have_metadata(self):
        """Test all losses have proper metadata."""
        loss_fns = get_losses()
        for name, expr in loss_fns.items():
            assert "category" in expr.metadata, f"{name} missing category"
            assert expr.metadata["category"] == "loss"
            assert "description" in expr.metadata, f"{name} missing description"

    def test_mse_zero(self):
        """Test MSE is zero when predictions match targets."""
        loss_fns = get_losses()
        mse = loss_fns["mse"]
        result = mse.eval(y_pred=1.0, y_true=1.0)
        assert result == 0.0

    def test_mse_positive(self):
        """Test MSE is positive for mismatched predictions."""
        loss_fns = get_losses()
        mse = loss_fns["mse"]
        result = mse.eval(y_pred=2.0, y_true=1.0)
        assert result == 1.0  # (2-1)^2

    def test_mae_zero(self):
        """Test MAE is zero when predictions match."""
        loss_fns = get_losses()
        mae = loss_fns["mae"]
        result = mae.eval(y_pred=1.0, y_true=1.0)
        assert result == 0.0

    def test_mae_positive(self):
        """Test MAE returns absolute difference."""
        loss_fns = get_losses()
        mae = loss_fns["mae"]
        result = mae.eval(y_pred=3.0, y_true=1.0)
        assert result == 2.0

    def test_hinge_correct(self):
        """Test hinge loss for correct classification."""
        loss_fns = get_losses()
        hinge = loss_fns["hinge"]
        # y_true=1, y_pred=2: max(0, 1 - 1*2) = max(0, -1) = 0
        result = hinge.eval(y_pred=2.0, y_true=1.0)
        assert result == 0.0

    def test_hinge_incorrect(self):
        """Test hinge loss for incorrect classification."""
        loss_fns = get_losses()
        hinge = loss_fns["hinge"]
        # y_true=1, y_pred=-1: max(0, 1 - 1*(-1)) = max(0, 2) = 2
        result = hinge.eval(y_pred=-1.0, y_true=1.0)
        assert result == 2.0
