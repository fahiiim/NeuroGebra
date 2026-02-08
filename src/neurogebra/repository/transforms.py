"""
Transforms expression repository.

Pre-built data transformation expressions: normalization, standardization,
feature engineering, encoding, and signal processing transforms commonly
used in ML preprocessing and feature pipelines.
"""

from neurogebra.core.expression import Expression
from typing import Dict


def get_transforms_expressions() -> Dict[str, Expression]:
    """
    Get dictionary of transform expressions.

    Returns:
        Dictionary mapping names to Expression instances
    """
    tf: Dict[str, Expression] = {}

    # ================================================================
    # Normalization Transforms
    # ================================================================

    tf["min_max_normalize"] = Expression(
        name="min_max_normalize",
        symbolic_expr="(x - x_min) / (x_max - x_min + 1e-8)",
        params={"x_min": 0, "x_max": 1},
        metadata={
            "category": "transform",
            "subcategory": "normalization",
            "description": "Min-Max normalization to [0, 1]",
            "usage": "Feature scaling, image pixel normalization",
            "formula_latex": r"\frac{x - x_{min}}{x_{max} - x_{min}}",
        },
    )

    tf["min_max_to_range"] = Expression(
        name="min_max_to_range",
        symbolic_expr="a_range + (x - x_min) * (b_range - a_range) / (x_max - x_min + 1e-8)",
        params={"x_min": 0, "x_max": 1, "a_range": -1, "b_range": 1},
        metadata={
            "category": "transform",
            "subcategory": "normalization",
            "description": "Min-Max normalization to [a, b]",
            "usage": "Scaling to arbitrary range (e.g., [-1, 1])",
            "formula_latex": r"a + \frac{(x - x_{min})(b - a)}{x_{max} - x_{min}}",
        },
    )

    tf["z_score_normalize"] = Expression(
        name="z_score_normalize",
        symbolic_expr="(x - mu) / (sigma + 1e-8)",
        params={"mu": 0, "sigma": 1},
        metadata={
            "category": "transform",
            "subcategory": "normalization",
            "description": "Z-score standardization (mean=0, std=1)",
            "usage": "Most common preprocessing, SVM, logistic regression",
            "formula_latex": r"\frac{x - \mu}{\sigma}",
        },
    )

    tf["robust_scale"] = Expression(
        name="robust_scale",
        symbolic_expr="(x - median_val) / (iqr + 1e-8)",
        params={"median_val": 0, "iqr": 1},
        metadata={
            "category": "transform",
            "subcategory": "normalization",
            "description": "Robust scaling (median and IQR)",
            "usage": "Data with outliers, robust preprocessing",
            "formula_latex": r"\frac{x - \text{median}}{IQR}",
        },
    )

    tf["l2_normalize"] = Expression(
        name="l2_normalize",
        symbolic_expr="x / (norm_l2 + 1e-8)",
        params={"norm_l2": 1},
        metadata={
            "category": "transform",
            "subcategory": "normalization",
            "description": "L2 (unit vector) normalization",
            "usage": "Embedding normalization, cosine similarity prep",
            "formula_latex": r"\frac{x}{\|x\|_2}",
        },
    )

    tf["max_abs_scale"] = Expression(
        name="max_abs_scale",
        symbolic_expr="x / (max_abs + 1e-8)",
        params={"max_abs": 1},
        metadata={
            "category": "transform",
            "subcategory": "normalization",
            "description": "MaxAbs scaling (preserves sparsity)",
            "usage": "Sparse data, data already centered at zero",
            "formula_latex": r"\frac{x}{\max|x|}",
        },
    )

    # ================================================================
    # Logarithmic & Power Transforms
    # ================================================================

    tf["log_transform"] = Expression(
        name="log_transform",
        symbolic_expr="log(x + 1)",
        metadata={
            "category": "transform",
            "subcategory": "power",
            "description": "Log(1+x) transform for right-skewed data",
            "usage": "Count data, income, prices",
            "formula_latex": r"\ln(x + 1)",
        },
    )

    tf["log10_transform"] = Expression(
        name="log10_transform",
        symbolic_expr="log(x + 1, 10)",
        metadata={
            "category": "transform",
            "subcategory": "power",
            "description": "Log base-10 transform",
            "usage": "Orders of magnitude, scientific data",
            "formula_latex": r"\log_{10}(x + 1)",
        },
    )

    tf["sqrt_transform"] = Expression(
        name="sqrt_transform",
        symbolic_expr="sqrt(x)",
        metadata={
            "category": "transform",
            "subcategory": "power",
            "description": "Square root transform (mild right-skew fix)",
            "usage": "Count data, variance stabilization",
            "formula_latex": r"\sqrt{x}",
        },
    )

    tf["box_cox_approx"] = Expression(
        name="box_cox_approx",
        symbolic_expr="Piecewise(((x**lambda_bc - 1) / lambda_bc, lambda_bc != 0), (log(x), True))",
        params={"lambda_bc": 0.5},
        metadata={
            "category": "transform",
            "subcategory": "power",
            "description": "Box-Cox transform (power family for normality)",
            "usage": "Making data more Gaussian, regression assumption",
            "formula_latex": r"\frac{x^\lambda - 1}{\lambda}",
        },
    )

    tf["yeo_johnson_positive"] = Expression(
        name="yeo_johnson_positive",
        symbolic_expr="((x + 1)**lambda_yj - 1) / lambda_yj",
        params={"lambda_yj": 0.5},
        metadata={
            "category": "transform",
            "subcategory": "power",
            "description": "Yeo-Johnson transform (positive x, λ≠0)",
            "usage": "Handles zero and negative values (unlike Box-Cox)",
            "formula_latex": r"\frac{(x+1)^\lambda - 1}{\lambda}",
        },
    )

    tf["power_transform"] = Expression(
        name="power_transform",
        symbolic_expr="x**p_exp",
        params={"p_exp": 2.0},
        metadata={
            "category": "transform",
            "subcategory": "power",
            "description": "General power transform x^p",
            "usage": "Feature engineering, polynomial features",
            "formula_latex": r"x^p",
        },
    )

    # ================================================================
    # Activation-Style Transforms (used in preprocessing)
    # ================================================================

    tf["sigmoid_transform"] = Expression(
        name="sigmoid_transform",
        symbolic_expr="1 / (1 + exp(-k_sig * (x - x0_sig)))",
        params={"k_sig": 1, "x0_sig": 0},
        metadata={
            "category": "transform",
            "subcategory": "nonlinear",
            "description": "Sigmoid squashing to (0, 1)",
            "usage": "Probability calibration, soft thresholding",
            "formula_latex": r"\frac{1}{1 + e^{-k(x-x_0)}}",
        },
    )

    tf["tanh_transform"] = Expression(
        name="tanh_transform",
        symbolic_expr="tanh(x)",
        metadata={
            "category": "transform",
            "subcategory": "nonlinear",
            "description": "Tanh squashing to (-1, 1)",
            "usage": "Centering with bounded output",
            "formula_latex": r"\tanh(x)",
        },
    )

    tf["softplus_transform"] = Expression(
        name="softplus_transform",
        symbolic_expr="log(1 + exp(x))",
        metadata={
            "category": "transform",
            "subcategory": "nonlinear",
            "description": "Softplus: smooth approximation to ReLU",
            "usage": "Ensuring positive outputs, smooth activation",
            "formula_latex": r"\ln(1 + e^x)",
        },
    )

    tf["softsign_transform"] = Expression(
        name="softsign_transform",
        symbolic_expr="x / (1 + Abs(x))",
        metadata={
            "category": "transform",
            "subcategory": "nonlinear",
            "description": "Softsign: slower saturation than tanh",
            "usage": "Alternative to tanh, polynomial decay",
            "formula_latex": r"\frac{x}{1 + |x|}",
        },
    )

    # ================================================================
    # Clipping & Quantization
    # ================================================================

    tf["clip_transform"] = Expression(
        name="clip_transform",
        symbolic_expr="Max(low, Min(x, high))",
        params={"low": 0, "high": 1},
        metadata={
            "category": "transform",
            "subcategory": "clipping",
            "description": "Clip (clamp) values to [low, high]",
            "usage": "Gradient clipping, pixel range enforcement",
            "formula_latex": r"\text{clip}(x, a, b)",
        },
    )

    tf["winsorize"] = Expression(
        name="winsorize",
        symbolic_expr="Max(lower_pct, Min(x, upper_pct))",
        params={"lower_pct": -3, "upper_pct": 3},
        metadata={
            "category": "transform",
            "subcategory": "clipping",
            "description": "Winsorize (clip to percentile bounds)",
            "usage": "Outlier handling, robust statistics",
            "formula_latex": r"\text{clip}(x, q_{lo}, q_{hi})",
        },
    )

    # ================================================================
    # Encoding Transforms
    # ================================================================

    tf["thermometer_bit"] = Expression(
        name="thermometer_bit",
        symbolic_expr="Piecewise((1, x >= threshold), (0, True))",
        params={"threshold": 0.5},
        metadata={
            "category": "transform",
            "subcategory": "encoding",
            "description": "Thermometer encoding bit (1 if x ≥ threshold)",
            "usage": "Thermometer encoding, binary discretization",
            "formula_latex": r"\mathbb{1}[x \ge \theta]",
        },
    )

    tf["gaussian_basis"] = Expression(
        name="gaussian_basis",
        symbolic_expr="exp(-((x - center)**2) / (2 * width**2))",
        params={"center": 0, "width": 1},
        trainable_params=["center", "width"],
        metadata={
            "category": "transform",
            "subcategory": "encoding",
            "description": "Gaussian radial basis function",
            "usage": "RBF feature encoding, kernel approximation",
            "formula_latex": r"e^{-\frac{(x-c)^2}{2w^2}}",
        },
    )

    tf["fourier_feature_sin"] = Expression(
        name="fourier_feature_sin",
        symbolic_expr="sin(2 * pi * freq * x)",
        params={"freq": 1},
        metadata={
            "category": "transform",
            "subcategory": "encoding",
            "description": "Sine component of random Fourier feature",
            "usage": "Random Fourier features, positional encoding",
            "formula_latex": r"\sin(2\pi f x)",
        },
    )

    tf["fourier_feature_cos"] = Expression(
        name="fourier_feature_cos",
        symbolic_expr="cos(2 * pi * freq * x)",
        params={"freq": 1},
        metadata={
            "category": "transform",
            "subcategory": "encoding",
            "description": "Cosine component of random Fourier feature",
            "usage": "Random Fourier features, positional encoding",
            "formula_latex": r"\cos(2\pi f x)",
        },
    )

    tf["positional_enc_sin"] = Expression(
        name="positional_enc_sin",
        symbolic_expr="sin(pos / (10000**(2*i / d_model)))",
        params={"pos": 1, "i": 0, "d_model": 512},
        metadata={
            "category": "transform",
            "subcategory": "encoding",
            "description": "Transformer positional encoding (sine component)",
            "usage": "Transformer input embeddings",
            "formula_latex": r"PE(pos,2i) = \sin\frac{pos}{10000^{2i/d}}",
        },
    )

    tf["positional_enc_cos"] = Expression(
        name="positional_enc_cos",
        symbolic_expr="cos(pos / (10000**(2*i / d_model)))",
        params={"pos": 1, "i": 0, "d_model": 512},
        metadata={
            "category": "transform",
            "subcategory": "encoding",
            "description": "Transformer positional encoding (cosine component)",
            "usage": "Transformer input embeddings",
            "formula_latex": r"PE(pos,2i+1) = \cos\frac{pos}{10000^{2i/d}}",
        },
    )

    # ================================================================
    # Signal Processing Transforms
    # ================================================================

    tf["moving_average_weight"] = Expression(
        name="moving_average_weight",
        symbolic_expr="1 / window_size",
        params={"window_size": 5},
        metadata={
            "category": "transform",
            "subcategory": "signal",
            "description": "Uniform moving average weight",
            "usage": "Smoothing time series, noise reduction",
            "formula_latex": r"\frac{1}{k}",
        },
    )

    tf["exponential_smoothing"] = Expression(
        name="exponential_smoothing",
        symbolic_expr="alpha_es * x + (1 - alpha_es) * s_prev",
        params={"alpha_es": 0.3, "s_prev": 0},
        metadata={
            "category": "transform",
            "subcategory": "signal",
            "description": "Exponential smoothing update",
            "usage": "Time series forecasting, EMA indicators",
            "formula_latex": r"s_t = \alpha x_t + (1-\alpha) s_{t-1}",
        },
    )

    tf["difference_transform"] = Expression(
        name="difference_transform",
        symbolic_expr="x - x_prev",
        params={"x_prev": 0},
        metadata={
            "category": "transform",
            "subcategory": "signal",
            "description": "First-order differencing",
            "usage": "Making time series stationary, detrending",
            "formula_latex": r"\Delta x_t = x_t - x_{t-1}",
        },
    )

    tf["log_return"] = Expression(
        name="log_return",
        symbolic_expr="log(x / (x_prev + 1e-8))",
        params={"x_prev": 1},
        metadata={
            "category": "transform",
            "subcategory": "signal",
            "description": "Log return (financial time series)",
            "usage": "Stock prices, financial modeling",
            "formula_latex": r"r_t = \ln\frac{P_t}{P_{t-1}}",
        },
    )

    # ================================================================
    # Weight Initialization Formulas
    # ================================================================

    tf["xavier_uniform_bound"] = Expression(
        name="xavier_uniform_bound",
        symbolic_expr="sqrt(6 / (fan_in + fan_out))",
        params={"fan_in": 256, "fan_out": 256},
        metadata={
            "category": "transform",
            "subcategory": "initialization",
            "description": "Xavier/Glorot uniform init bound",
            "usage": "Linear/sigmoid layer initialization",
            "formula_latex": r"\text{bound} = \sqrt{\frac{6}{n_{in}+n_{out}}}",
        },
    )

    tf["xavier_normal_std"] = Expression(
        name="xavier_normal_std",
        symbolic_expr="sqrt(2 / (fan_in + fan_out))",
        params={"fan_in": 256, "fan_out": 256},
        metadata={
            "category": "transform",
            "subcategory": "initialization",
            "description": "Xavier/Glorot normal init standard deviation",
            "usage": "Linear/sigmoid layer initialization",
            "formula_latex": r"\sigma = \sqrt{\frac{2}{n_{in}+n_{out}}}",
        },
    )

    tf["he_uniform_bound"] = Expression(
        name="he_uniform_bound",
        symbolic_expr="sqrt(6 / fan_in)",
        params={"fan_in": 256},
        metadata={
            "category": "transform",
            "subcategory": "initialization",
            "description": "He/Kaiming uniform init bound",
            "usage": "ReLU layer initialization",
            "formula_latex": r"\text{bound} = \sqrt{\frac{6}{n_{in}}}",
        },
    )

    tf["he_normal_std"] = Expression(
        name="he_normal_std",
        symbolic_expr="sqrt(2 / fan_in)",
        params={"fan_in": 256},
        metadata={
            "category": "transform",
            "subcategory": "initialization",
            "description": "He/Kaiming normal init standard deviation",
            "usage": "ReLU layer initialization",
            "formula_latex": r"\sigma = \sqrt{\frac{2}{n_{in}}}",
        },
    )

    tf["lecun_normal_std"] = Expression(
        name="lecun_normal_std",
        symbolic_expr="sqrt(1 / fan_in)",
        params={"fan_in": 256},
        metadata={
            "category": "transform",
            "subcategory": "initialization",
            "description": "LeCun normal init standard deviation",
            "usage": "SELU activation, self-normalizing networks",
            "formula_latex": r"\sigma = \sqrt{\frac{1}{n_{in}}}",
        },
    )

    return tf
