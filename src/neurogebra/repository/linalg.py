"""
Linear algebra expression repository.

Pre-built linear algebra expressions: norms, distances, inner products,
projections, and matrix operation components commonly used in ML.
"""

from neurogebra.core.expression import Expression
from typing import Dict


def get_linalg_expressions() -> Dict[str, Expression]:
    """
    Get dictionary of linear algebra expressions.

    Returns:
        Dictionary mapping names to Expression instances
    """
    la: Dict[str, Expression] = {}

    # ================================================================
    # Vector Norms (element-wise components)
    # ================================================================

    la["l1_norm_elem"] = Expression(
        name="l1_norm_elem",
        symbolic_expr="Abs(x)",
        metadata={
            "category": "linalg",
            "subcategory": "norm",
            "description": "L1 norm element |xᵢ| (sum over elements for vector norm)",
            "usage": "Sparsity, Manhattan distance, LASSO",
            "formula_latex": r"|x_i|",
        },
    )

    la["l2_norm_elem"] = Expression(
        name="l2_norm_elem",
        symbolic_expr="x**2",
        metadata={
            "category": "linalg",
            "subcategory": "norm",
            "description": "L2 norm element xᵢ² (sum then sqrt for vector norm)",
            "usage": "Euclidean distance, weight decay, Ridge",
            "formula_latex": r"x_i^2",
        },
    )

    la["lp_norm_elem"] = Expression(
        name="lp_norm_elem",
        symbolic_expr="Abs(x)**p",
        params={"p": 2},
        metadata={
            "category": "linalg",
            "subcategory": "norm",
            "description": "General Lp norm element |xᵢ|^p",
            "usage": "Generalized norms, robust statistics",
            "formula_latex": r"|x_i|^p",
        },
    )

    la["huber_norm_elem"] = Expression(
        name="huber_norm_elem",
        symbolic_expr="Piecewise((x**2 / 2, Abs(x) <= delta), (delta * (Abs(x) - delta / 2), True))",
        params={"delta": 1},
        metadata={
            "category": "linalg",
            "subcategory": "norm",
            "description": "Huber norm element (smooth L1/L2 transition)",
            "usage": "Robust regression, smooth optimization",
            "formula_latex": r"\begin{cases} x^2/2 & |x|\le\delta \\ \delta(|x|-\delta/2) & \text{otherwise} \end{cases}",
        },
    )

    # ================================================================
    # Inner Products & Similarities
    # ================================================================

    la["dot_product_elem"] = Expression(
        name="dot_product_elem",
        symbolic_expr="x * y",
        metadata={
            "category": "linalg",
            "subcategory": "inner_product",
            "description": "Dot product element xᵢ·yᵢ (sum for full dot product)",
            "usage": "Similarity, projections, attention scores",
            "formula_latex": r"x_i \cdot y_i",
        },
    )

    la["cosine_similarity"] = Expression(
        name="cosine_similarity",
        symbolic_expr="dot_xy / (norm_x * norm_y + 1e-8)",
        params={"dot_xy": 1, "norm_x": 1, "norm_y": 1},
        metadata={
            "category": "linalg",
            "subcategory": "similarity",
            "description": "Cosine similarity cos(θ) = (x·y) / (‖x‖‖y‖)",
            "usage": "Text similarity, recommendation, embeddings",
            "formula_latex": r"\cos\theta = \frac{\mathbf{x}\cdot\mathbf{y}}{\|\mathbf{x}\|\|\mathbf{y}\|}",
        },
    )

    la["scaled_dot_product"] = Expression(
        name="scaled_dot_product",
        symbolic_expr="dot_qk / sqrt(d_k)",
        params={"dot_qk": 1, "d_k": 64},
        metadata={
            "category": "linalg",
            "subcategory": "inner_product",
            "description": "Scaled dot-product (Transformer attention core)",
            "usage": "Self-attention, cross-attention in Transformers",
            "formula_latex": r"\frac{Q \cdot K^T}{\sqrt{d_k}}",
        },
    )

    # ================================================================
    # Distance Metrics
    # ================================================================

    la["euclidean_dist_elem"] = Expression(
        name="euclidean_dist_elem",
        symbolic_expr="(x - y)**2",
        metadata={
            "category": "linalg",
            "subcategory": "distance",
            "description": "Squared Euclidean distance element (sum then sqrt)",
            "usage": "K-means, KNN, distance-based clustering",
            "formula_latex": r"(x_i - y_i)^2",
        },
    )

    la["manhattan_dist_elem"] = Expression(
        name="manhattan_dist_elem",
        symbolic_expr="Abs(x - y)",
        metadata={
            "category": "linalg",
            "subcategory": "distance",
            "description": "Manhattan distance element |xᵢ - yᵢ|",
            "usage": "High-dimensional data, grid movement",
            "formula_latex": r"|x_i - y_i|",
        },
    )

    la["minkowski_dist_elem"] = Expression(
        name="minkowski_dist_elem",
        symbolic_expr="Abs(x - y)**p",
        params={"p": 2},
        metadata={
            "category": "linalg",
            "subcategory": "distance",
            "description": "Minkowski distance element |xᵢ - yᵢ|^p",
            "usage": "Generalized distance (p=1 Manhattan, p=2 Euclidean)",
            "formula_latex": r"|x_i - y_i|^p",
        },
    )

    la["chebyshev_dist"] = Expression(
        name="chebyshev_dist",
        symbolic_expr="Max(Abs(x - y), Abs(a - b))",
        metadata={
            "category": "linalg",
            "subcategory": "distance",
            "description": "Chebyshev distance max|xᵢ - yᵢ| (2D demo)",
            "usage": "Chess-board distance, L∞ norm",
            "formula_latex": r"\max_i |x_i - y_i|",
        },
    )

    la["mahalanobis_1d"] = Expression(
        name="mahalanobis_1d",
        symbolic_expr="Abs(x - mu) / sigma",
        params={"mu": 0, "sigma": 1},
        metadata={
            "category": "linalg",
            "subcategory": "distance",
            "description": "1D Mahalanobis distance (generalized z-score)",
            "usage": "Outlier detection, multivariate anomaly detection",
            "formula_latex": r"\frac{|x - \mu|}{\sigma}",
        },
    )

    la["canberra_dist_elem"] = Expression(
        name="canberra_dist_elem",
        symbolic_expr="Abs(x - y) / (Abs(x) + Abs(y) + 1e-8)",
        metadata={
            "category": "linalg",
            "subcategory": "distance",
            "description": "Canberra distance element",
            "usage": "Sensitive to small values, ecology data",
            "formula_latex": r"\frac{|x_i - y_i|}{|x_i| + |y_i|}",
        },
    )

    # ================================================================
    # Projections & Decompositions
    # ================================================================

    la["vector_projection_scalar"] = Expression(
        name="vector_projection_scalar",
        symbolic_expr="dot_ab / (norm_b**2 + 1e-8)",
        params={"dot_ab": 1, "norm_b": 1},
        metadata={
            "category": "linalg",
            "subcategory": "projection",
            "description": "Scalar projection coefficient (a·b / ‖b‖²)",
            "usage": "Gram-Schmidt, PCA components, projections",
            "formula_latex": r"\frac{\mathbf{a}\cdot\mathbf{b}}{\|\mathbf{b}\|^2}",
        },
    )

    la["orthogonal_residual_elem"] = Expression(
        name="orthogonal_residual_elem",
        symbolic_expr="a - proj_coeff * b",
        params={"proj_coeff": 0},
        metadata={
            "category": "linalg",
            "subcategory": "projection",
            "description": "Orthogonal residual element aᵢ - (proj coeff)·bᵢ",
            "usage": "Gram-Schmidt orthogonalization, residual computation",
            "formula_latex": r"a_i - \frac{\mathbf{a}\cdot\mathbf{b}}{\|\mathbf{b}\|^2} b_i",
        },
    )

    # ================================================================
    # Matrix Operations (2×2 scalar forms)
    # ================================================================

    la["determinant_2x2"] = Expression(
        name="determinant_2x2",
        symbolic_expr="a * d - b * c",
        metadata={
            "category": "linalg",
            "subcategory": "matrix",
            "description": "Determinant of 2×2 matrix [[a,b],[c,d]]",
            "usage": "Invertibility, area scaling, eigenvalue computation",
            "formula_latex": r"\det = ad - bc",
        },
    )

    la["trace_2x2"] = Expression(
        name="trace_2x2",
        symbolic_expr="a + d",
        metadata={
            "category": "linalg",
            "subcategory": "matrix",
            "description": "Trace of 2×2 matrix [[a,b],[c,d]]",
            "usage": "Sum of eigenvalues, matrix regularization",
            "formula_latex": r"\text{tr}(A) = a + d",
        },
    )

    la["eigenvalue_2x2"] = Expression(
        name="eigenvalue_2x2",
        symbolic_expr="(a + d) / 2 + sqrt(((a - d) / 2)**2 + b * c)",
        metadata={
            "category": "linalg",
            "subcategory": "matrix",
            "description": "Larger eigenvalue of 2×2 matrix [[a,b],[c,d]]",
            "usage": "PCA, spectral analysis, stability analysis",
            "formula_latex": r"\lambda_1 = \frac{a+d}{2} + \sqrt{(\frac{a-d}{2})^2 + bc}",
        },
    )

    la["frobenius_elem"] = Expression(
        name="frobenius_elem",
        symbolic_expr="x**2",
        metadata={
            "category": "linalg",
            "subcategory": "matrix",
            "description": "Frobenius norm element (sum all aᵢⱼ², then sqrt)",
            "usage": "Matrix norm, regularization of weight matrices",
            "formula_latex": r"a_{ij}^2",
        },
    )

    # ================================================================
    # Kernel Functions (additional kernels; see also algebra.py)
    # ================================================================

    la["sigmoid_kernel"] = Expression(
        name="sigmoid_kernel",
        symbolic_expr="tanh(alpha * x * y + c_coeff)",
        params={"alpha": 1, "c_coeff": 0},
        metadata={
            "category": "linalg",
            "subcategory": "kernel",
            "description": "Sigmoid (hyperbolic tangent) kernel",
            "usage": "Neural network-inspired kernel, SVM",
            "formula_latex": r"K(x,y) = \tanh(\alpha x y + c)",
        },
    )

    # ================================================================
    # Attention & Transformer Components
    # ================================================================

    la["softmax_score"] = Expression(
        name="softmax_score",
        symbolic_expr="exp(x) / (exp(x) + exp(y) + 1e-8)",
        metadata={
            "category": "linalg",
            "subcategory": "attention",
            "description": "2-class softmax probability",
            "usage": "Attention weights, classification head",
            "formula_latex": r"\frac{e^{x_i}}{\sum_j e^{x_j}}",
        },
    )

    la["log_softmax_score"] = Expression(
        name="log_softmax_score",
        symbolic_expr="x - log(exp(x) + exp(y) + 1e-8)",
        metadata={
            "category": "linalg",
            "subcategory": "attention",
            "description": "Log-softmax (numerically stable, 2-class)",
            "usage": "NLLLoss input, cross-entropy computation",
            "formula_latex": r"x_i - \ln\sum_j e^{x_j}",
        },
    )

    la["layer_norm_elem"] = Expression(
        name="layer_norm_elem",
        symbolic_expr="gamma_ln * (x - mu) / (sigma + 1e-5) + beta_ln",
        params={"mu": 0, "sigma": 1, "gamma_ln": 1, "beta_ln": 0},
        trainable_params=["gamma_ln", "beta_ln"],
        metadata={
            "category": "linalg",
            "subcategory": "normalization",
            "description": "Layer normalization element",
            "usage": "Transformers, stabilizing training",
            "formula_latex": r"\gamma \frac{x - \mu}{\sigma + \epsilon} + \beta",
        },
    )

    la["batch_norm_elem"] = Expression(
        name="batch_norm_elem",
        symbolic_expr="gamma_bn * (x - mu_batch) / (sqrt(var_batch + 1e-5)) + beta_bn",
        params={"mu_batch": 0, "var_batch": 1, "gamma_bn": 1, "beta_bn": 0},
        trainable_params=["gamma_bn", "beta_bn"],
        metadata={
            "category": "linalg",
            "subcategory": "normalization",
            "description": "Batch normalization element",
            "usage": "CNNs, stabilizing deep network training",
            "formula_latex": r"\gamma \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} + \beta",
        },
    )

    return la
