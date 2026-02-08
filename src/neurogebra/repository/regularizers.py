"""
Regularizer repository.

Pre-built regularization expressions for neural network training.
Regularizers add penalty terms to the loss to prevent overfitting.
"""

from neurogebra.core.expression import Expression
from typing import Dict


def get_regularizers() -> Dict[str, Expression]:
    """
    Get dictionary of regularization expressions.

    Returns:
        Dictionary mapping names to Expression instances
    """
    regs: Dict[str, Expression] = {}

    # ----------------------------------------------------------------
    # Classic Norm-Based Regularizers
    # ----------------------------------------------------------------

    regs["l1"] = Expression(
        name="l1",
        symbolic_expr="lambda_reg * Abs(w)",
        params={"lambda_reg": 0.01},
        metadata={
            "category": "regularizer",
            "description": "L1 regularization (Lasso) - promotes sparsity",
            "usage": "Feature selection, sparse models",
            "pros": ["Promotes sparsity", "Feature selection"],
            "cons": ["Not differentiable at zero"],
            "formula_latex": r"\lambda |w|",
        },
    )

    regs["l2"] = Expression(
        name="l2",
        symbolic_expr="lambda_reg * w**2",
        params={"lambda_reg": 0.01},
        metadata={
            "category": "regularizer",
            "description": "L2 regularization (Ridge/Weight Decay)",
            "usage": "Prevent overfitting, weight decay",
            "pros": ["Smooth", "Well-understood", "Convex"],
            "cons": ["Does not promote sparsity"],
            "formula_latex": r"\lambda w^2",
        },
    )

    regs["elastic_net"] = Expression(
        name="elastic_net",
        symbolic_expr=(
            "alpha * lambda_reg * Abs(w) "
            "+ (1 - alpha) * lambda_reg * w**2"
        ),
        params={"lambda_reg": 0.01, "alpha": 0.5},
        metadata={
            "category": "regularizer",
            "description": "Elastic Net - combination of L1 and L2",
            "usage": "When both sparsity and grouping are desired",
            "pros": ["Combines L1 and L2 benefits"],
            "cons": ["Extra hyperparameter"],
        },
    )

    regs["weight_decay"] = Expression(
        name="weight_decay",
        symbolic_expr="0.5 * lambda_reg * w**2",
        params={"lambda_reg": 0.01},
        metadata={
            "category": "regularizer",
            "description": "Weight decay - direct weight shrinkage (½λw²)",
            "usage": "AdamW optimizer, modern training",
            "pros": ["Decoupled from loss", "Stable training"],
            "cons": ["Equivalent to L2 for SGD"],
        },
    )

    regs["l1_l2"] = Expression(
        name="l1_l2",
        symbolic_expr="l1_reg * Abs(w) + l2_reg * w**2",
        params={"l1_reg": 0.01, "l2_reg": 0.01},
        metadata={
            "category": "regularizer",
            "description": "Independent L1 + L2 with separate coefficients",
            "usage": "Fine-grained control over sparsity & smoothness",
            "pros": ["Independent tuning of each term"],
            "cons": ["Two hyperparameters"],
        },
    )

    # ----------------------------------------------------------------
    # Smooth Sparsity Regularizers
    # ----------------------------------------------------------------

    regs["log_barrier"] = Expression(
        name="log_barrier",
        symbolic_expr="-lambda_reg * log(1 - w**2 + epsilon)",
        params={"lambda_reg": 0.01, "epsilon": 1e-8},
        metadata={
            "category": "regularizer",
            "description": "Log-barrier penalty - keeps weights bounded",
            "usage": "Constrained optimization, bounded weights",
            "pros": ["Smooth", "Enforces weight bounds"],
            "cons": ["Undefined outside bounds"],
        },
    )

    regs["sqrt_reg"] = Expression(
        name="sqrt_reg",
        symbolic_expr="lambda_reg * sqrt(w**2 + epsilon)",
        params={"lambda_reg": 0.01, "epsilon": 1e-8},
        metadata={
            "category": "regularizer",
            "description": "Smooth L1 approximation via √(w² + ε)",
            "usage": "Differentiable sparsity, compressed sensing",
            "pros": ["Smooth everywhere", "Approximates L1"],
            "cons": ["Less sparse than true L1"],
        },
    )

    regs["cauchy_reg"] = Expression(
        name="cauchy_reg",
        symbolic_expr="lambda_reg * log(1 + (w / sigma)**2)",
        params={"lambda_reg": 0.01, "sigma": 1.0},
        metadata={
            "category": "regularizer",
            "description": "Cauchy (Lorentzian) penalty - robust sparsity",
            "usage": "Robust regression, outlier-tolerant sparsity",
            "pros": ["Non-convex", "Strongly promotes sparsity"],
            "cons": ["Non-convex, harder to optimize"],
        },
    )

    # ----------------------------------------------------------------
    # Information-Theoretic Regularizers
    # ----------------------------------------------------------------

    regs["entropy_reg"] = Expression(
        name="entropy_reg",
        symbolic_expr=(
            "-lambda_reg * (p * log(p + epsilon) "
            "+ (1 - p) * log(1 - p + epsilon))"
        ),
        params={"lambda_reg": 0.01, "epsilon": 1e-8},
        metadata={
            "category": "regularizer",
            "description": "Entropy regularization - encourages confidence",
            "usage": "Reinforcement learning, semi-supervised learning",
            "pros": ["Encourages confident predictions"],
            "cons": ["Can reduce exploration"],
        },
    )

    regs["kl_divergence_reg"] = Expression(
        name="kl_divergence_reg",
        symbolic_expr=(
            "lambda_reg * (p * log((p + epsilon)/(q + epsilon)) "
            "+ (1 - p) * log((1 - p + epsilon)/(1 - q + epsilon)))"
        ),
        params={"lambda_reg": 0.01, "epsilon": 1e-8},
        metadata={
            "category": "regularizer",
            "description": "KL divergence penalty - keep distribution close to prior",
            "usage": "Variational autoencoders (VAE), Bayesian learning",
            "pros": ["Principled Bayesian regularization"],
            "cons": ["Asymmetric", "Requires prior q"],
        },
    )

    # ----------------------------------------------------------------
    # Gradient / Smoothness Regularizers
    # ----------------------------------------------------------------

    regs["gradient_penalty"] = Expression(
        name="gradient_penalty",
        symbolic_expr="lambda_reg * (grad_norm - 1)**2",
        params={"lambda_reg": 10.0},
        metadata={
            "category": "regularizer",
            "description": "Gradient penalty - enforce Lipschitz constraint",
            "usage": "WGAN-GP, Lipschitz-constrained networks",
            "pros": ["Stabilizes GAN training", "Enforces smoothness"],
            "cons": ["Expensive to compute"],
        },
    )

    regs["total_variation"] = Expression(
        name="total_variation",
        symbolic_expr="lambda_reg * Abs(w_i - w_j)",
        params={"lambda_reg": 0.01},
        metadata={
            "category": "regularizer",
            "description": "Total variation - promote spatial smoothness",
            "usage": "Image denoising, signal smoothing",
            "pros": ["Preserves edges", "Removes noise"],
            "cons": ["Staircase artifacts"],
        },
    )

    regs["tikhonov"] = Expression(
        name="tikhonov",
        symbolic_expr="lambda_reg * (w - w_prior)**2",
        params={"lambda_reg": 0.01, "w_prior": 0.0},
        metadata={
            "category": "regularizer",
            "description": "Tikhonov regularization - shrink toward prior",
            "usage": "Ill-posed inverse problems, Bayesian MAP",
            "pros": ["Principled", "Centers weights around prior"],
            "cons": ["Requires choosing prior"],
        },
    )

    # ----------------------------------------------------------------
    # Sparsity-Inducing Penalty Functions
    # ----------------------------------------------------------------

    regs["group_lasso"] = Expression(
        name="group_lasso",
        symbolic_expr="lambda_reg * sqrt(w1**2 + w2**2 + epsilon)",
        params={"lambda_reg": 0.01, "epsilon": 1e-8},
        metadata={
            "category": "regularizer",
            "description": "Group Lasso - sets entire feature groups to zero",
            "usage": "Feature group selection, structured sparsity",
            "pros": ["Structured sparsity", "Group selection"],
            "cons": ["Requires grouping definition"],
        },
    )

    regs["scad"] = Expression(
        name="scad",
        symbolic_expr=(
            "Piecewise("
            "(lambda_reg * Abs(w), Abs(w) <= lambda_reg), "
            "(-(w**2 - 2*a*lambda_reg*Abs(w) + lambda_reg**2) "
            "/ (2*(a - 1)), Abs(w) <= a*lambda_reg), "
            "(lambda_reg**2 * (a + 1) / 2, True)"
            ")"
        ),
        params={"lambda_reg": 0.01, "a": 3.7},
        metadata={
            "category": "regularizer",
            "description": "SCAD penalty - Smoothly Clipped Absolute Deviation",
            "usage": "Variable selection, unbiased estimation",
            "pros": ["Unbiased for large weights", "Oracle property"],
            "cons": ["Non-convex", "Extra hyperparameter a"],
        },
    )

    regs["mcp"] = Expression(
        name="mcp",
        symbolic_expr=(
            "Piecewise("
            "(lambda_reg * Abs(w) - w**2 / (2*gamma_mcp), "
            "Abs(w) <= gamma_mcp*lambda_reg), "
            "(gamma_mcp*lambda_reg**2 / 2, True)"
            ")"
        ),
        params={"lambda_reg": 0.01, "gamma_mcp": 3.0},
        metadata={
            "category": "regularizer",
            "description": "Minimax Concave Penalty (MCP)",
            "usage": "High-dimensional variable selection",
            "pros": ["Nearly unbiased", "Sparser than Lasso"],
            "cons": ["Non-convex", "Harder optimization"],
        },
    )

    # ----------------------------------------------------------------
    # Other Useful Regularizers
    # ----------------------------------------------------------------

    regs["max_norm"] = Expression(
        name="max_norm",
        symbolic_expr="Max(0, w**2 - max_val**2)",
        params={"max_val": 3.0},
        metadata={
            "category": "regularizer",
            "description": "Max-norm constraint penalty",
            "usage": "Dropout companion, bounded activations",
            "pros": ["Bounds weight magnitude", "Works well with dropout"],
            "cons": ["Non-smooth at boundary"],
        },
    )

    regs["orthogonal_reg"] = Expression(
        name="orthogonal_reg",
        symbolic_expr="lambda_reg * (w1*w2)**2",
        params={"lambda_reg": 0.01},
        metadata={
            "category": "regularizer",
            "description": "Orthogonal regularization - encourage decorrelated weights",
            "usage": "Improving gradient flow, preventing mode collapse",
            "pros": ["Better gradient flow", "Decorrelated features"],
            "cons": ["Pairwise computation O(n²)"],
        },
    )

    regs["label_smoothing"] = Expression(
        name="label_smoothing",
        symbolic_expr="(1 - epsilon_smooth) * y + epsilon_smooth / K",
        params={"epsilon_smooth": 0.1, "K": 10},
        metadata={
            "category": "regularizer",
            "description": "Label smoothing - soften hard targets",
            "usage": "Classification, knowledge distillation",
            "pros": ["Prevents overconfidence", "Improves calibration"],
            "cons": ["Slightly lower peak accuracy"],
        },
    )

    regs["confidence_penalty"] = Expression(
        name="confidence_penalty",
        symbolic_expr="-lambda_reg * p * log(p + epsilon)",
        params={"lambda_reg": 0.1, "epsilon": 1e-8},
        metadata={
            "category": "regularizer",
            "description": "Confidence penalty - discourages overconfident outputs",
            "usage": "Well-calibrated models, uncertainty estimation",
            "pros": ["Better calibrated probabilities"],
            "cons": ["Slightly lower accuracy"],
        },
    )

    return regs
