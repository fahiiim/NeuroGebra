"""
Loss function repository.

Pre-built loss functions for training neural networks.
"""

from neurogebra.core.expression import Expression
from typing import Dict


def get_losses() -> Dict[str, Expression]:
    """
    Get dictionary of loss function expressions.

    Returns:
        Dictionary mapping names to Expression instances
    """
    loss_fns: Dict[str, Expression] = {}

    # Mean Squared Error
    loss_fns["mse"] = Expression(
        name="mse",
        symbolic_expr="(y_pred - y_true)**2",
        metadata={
            "category": "loss",
            "description": "Mean Squared Error - standard regression loss",
            "usage": "Regression tasks",
            "pros": ["Smooth and convex", "Well-understood gradients"],
            "cons": ["Sensitive to outliers"],
        },
    )

    # Mean Absolute Error
    loss_fns["mae"] = Expression(
        name="mae",
        symbolic_expr="Abs(y_pred - y_true)",
        metadata={
            "category": "loss",
            "description": "Mean Absolute Error - robust regression loss",
            "usage": "Regression with outliers",
            "pros": ["Robust to outliers", "Interpretable"],
            "cons": ["Not differentiable at zero", "Slower convergence"],
        },
    )

    # Binary Cross-Entropy
    loss_fns["binary_crossentropy"] = Expression(
        name="binary_crossentropy",
        symbolic_expr="-y_true*log(y_pred) - (1-y_true)*log(1-y_pred)",
        metadata={
            "category": "loss",
            "description": "Binary cross-entropy for binary classification",
            "usage": "Binary classification with sigmoid output",
            "pros": ["Probabilistic interpretation", "Strong gradients"],
            "cons": ["Requires probability inputs (0, 1)"],
        },
    )

    # Huber Loss
    loss_fns["huber"] = Expression(
        name="huber",
        symbolic_expr=(
            "Piecewise("
            "(0.5*(y_pred - y_true)**2, Abs(y_pred - y_true) <= delta), "
            "(delta*Abs(y_pred - y_true) - 0.5*delta**2, True)"
            ")"
        ),
        params={"delta": 1.0},
        metadata={
            "category": "loss",
            "description": "Huber loss - MSE for small errors, MAE for large errors",
            "usage": "Robust regression, reinforcement learning",
            "pros": ["Best of MSE and MAE", "Differentiable everywhere"],
            "cons": ["Extra hyperparameter (delta)"],
        },
    )

    # Log-Cosh Loss
    loss_fns["log_cosh"] = Expression(
        name="log_cosh",
        symbolic_expr="log(cosh(y_pred - y_true))",
        metadata={
            "category": "loss",
            "description": "Logarithm of hyperbolic cosine loss",
            "usage": "Smooth alternative to Huber loss",
            "pros": ["Smooth", "Approximately MSE for small errors"],
            "cons": ["Less commonly used"],
        },
    )

    # Hinge Loss
    loss_fns["hinge"] = Expression(
        name="hinge",
        symbolic_expr="Max(0, 1 - y_true * y_pred)",
        metadata={
            "category": "loss",
            "description": "Hinge loss for SVM-style classification",
            "usage": "Support vector machines, maximum margin classifiers",
            "pros": ["Encourages margin", "Sparse solutions"],
            "cons": ["Not differentiable at hinge point"],
        },
    )

    # Squared Hinge Loss
    loss_fns["squared_hinge"] = Expression(
        name="squared_hinge",
        symbolic_expr="Max(0, 1 - y_true * y_pred)**2",
        metadata={
            "category": "loss",
            "description": "Squared hinge loss - smoother version of hinge",
            "usage": "Smooth SVM classification",
            "pros": ["Smooth", "Differentiable"],
            "cons": ["Penalizes outliers more"],
        },
    )

    # Quantile Loss
    loss_fns["quantile"] = Expression(
        name="quantile",
        symbolic_expr=(
            "Piecewise("
            "(q * (y_true - y_pred), y_true >= y_pred), "
            "((1 - q) * (y_pred - y_true), True)"
            ")"
        ),
        params={"q": 0.5},
        metadata={
            "category": "loss",
            "description": "Quantile loss for quantile regression",
            "usage": "Predicting specific quantiles of distribution",
            "pros": ["Asymmetric penalty", "Quantile estimation"],
            "cons": ["Not differentiable at zero"],
        },
    )

    return loss_fns
