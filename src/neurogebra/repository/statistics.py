"""
Statistics expression repository.

Pre-built statistical expressions: probability distributions, moments,
information theory, hypothesis testing, and descriptive statistics.
"""

from neurogebra.core.expression import Expression
from typing import Dict


def get_statistics_expressions() -> Dict[str, Expression]:
    """
    Get dictionary of statistical expressions.

    Returns:
        Dictionary mapping names to Expression instances
    """
    stats: Dict[str, Expression] = {}

    # ================================================================
    # Probability Density Functions
    # ================================================================

    stats["normal_pdf"] = Expression(
        name="normal_pdf",
        symbolic_expr="(1 / (sigma * sqrt(2 * pi))) * exp(-(x - mu)**2 / (2 * sigma**2))",
        params={"mu": 0, "sigma": 1},
        trainable_params=["mu", "sigma"],
        metadata={
            "category": "statistics",
            "subcategory": "distribution",
            "description": "Normal (Gaussian) probability density function",
            "usage": "Central limit theorem, Bayesian priors, noise modeling",
            "formula_latex": r"\frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}",
        },
    )

    stats["standard_normal_pdf"] = Expression(
        name="standard_normal_pdf",
        symbolic_expr="(1 / sqrt(2 * pi)) * exp(-x**2 / 2)",
        metadata={
            "category": "statistics",
            "subcategory": "distribution",
            "description": "Standard normal PDF (μ=0, σ=1)",
            "usage": "Z-scores, standard statistics",
            "formula_latex": r"\frac{1}{\sqrt{2\pi}} e^{-x^2/2}",
        },
    )

    stats["uniform_pdf"] = Expression(
        name="uniform_pdf",
        symbolic_expr="1 / (b - a)",
        params={"a": 0, "b": 1},
        metadata={
            "category": "statistics",
            "subcategory": "distribution",
            "description": "Uniform distribution PDF on [a, b]",
            "usage": "Random initialization, prior distributions",
            "formula_latex": r"\frac{1}{b - a}",
        },
    )

    stats["exponential_pdf"] = Expression(
        name="exponential_pdf",
        symbolic_expr="lambda_param * exp(-lambda_param * x)",
        params={"lambda_param": 1},
        trainable_params=["lambda_param"],
        metadata={
            "category": "statistics",
            "subcategory": "distribution",
            "description": "Exponential distribution PDF",
            "usage": "Waiting times, survival analysis, Poisson processes",
            "formula_latex": r"\lambda e^{-\lambda x}",
        },
    )

    stats["laplace_pdf"] = Expression(
        name="laplace_pdf",
        symbolic_expr="(1 / (2 * b)) * exp(-Abs(x - mu) / b)",
        params={"mu": 0, "b": 1},
        trainable_params=["mu", "b"],
        metadata={
            "category": "statistics",
            "subcategory": "distribution",
            "description": "Laplace distribution PDF",
            "usage": "Robust statistics, L1 regularization prior",
            "formula_latex": r"\frac{1}{2b} e^{-|x-\mu|/b}",
        },
    )

    stats["cauchy_pdf"] = Expression(
        name="cauchy_pdf",
        symbolic_expr="1 / (pi * gamma_param * (1 + ((x - x0) / gamma_param)**2))",
        params={"x0": 0, "gamma_param": 1},
        trainable_params=["x0", "gamma_param"],
        metadata={
            "category": "statistics",
            "subcategory": "distribution",
            "description": "Cauchy (Lorentzian) distribution PDF",
            "usage": "Heavy-tailed modeling, spectral lines",
            "formula_latex": r"\frac{1}{\pi\gamma[1+(\frac{x-x_0}{\gamma})^2]}",
        },
    )

    stats["log_normal_pdf"] = Expression(
        name="log_normal_pdf",
        symbolic_expr="(1 / (x * sigma * sqrt(2 * pi))) * exp(-(log(x) - mu)**2 / (2 * sigma**2))",
        params={"mu": 0, "sigma": 1},
        trainable_params=["mu", "sigma"],
        metadata={
            "category": "statistics",
            "subcategory": "distribution",
            "description": "Log-normal distribution PDF",
            "usage": "Income distributions, stock prices, multiplicative processes",
            "formula_latex": r"\frac{1}{x\sigma\sqrt{2\pi}} e^{-\frac{(\ln x - \mu)^2}{2\sigma^2}}",
        },
    )

    stats["rayleigh_pdf"] = Expression(
        name="rayleigh_pdf",
        symbolic_expr="(x / sigma**2) * exp(-x**2 / (2 * sigma**2))",
        params={"sigma": 1},
        trainable_params=["sigma"],
        metadata={
            "category": "statistics",
            "subcategory": "distribution",
            "description": "Rayleigh distribution PDF",
            "usage": "Wind speed modeling, signal amplitude",
            "formula_latex": r"\frac{x}{\sigma^2} e^{-x^2/(2\sigma^2)}",
        },
    )

    stats["gumbel_pdf"] = Expression(
        name="gumbel_pdf",
        symbolic_expr="(1 / beta_param) * exp(-(z + exp(-z)))",
        params={"mu": 0, "beta_param": 1},
        metadata={
            "category": "statistics",
            "subcategory": "distribution",
            "description": "Gumbel distribution PDF (z = (x-μ)/β)",
            "usage": "Extreme value theory, max/min modeling",
            "note": "z = (x - mu) / beta; substitute before eval",
            "formula_latex": r"\frac{1}{\beta} e^{-(z + e^{-z})}",
        },
    )

    stats["beta_pdf"] = Expression(
        name="beta_pdf",
        symbolic_expr="x**(alpha - 1) * (1 - x)**(beta_param - 1)",
        params={"alpha": 2, "beta_param": 5},
        trainable_params=["alpha", "beta_param"],
        metadata={
            "category": "statistics",
            "subcategory": "distribution",
            "description": "Beta distribution PDF (unnormalized, x ∈ [0,1])",
            "usage": "Bayesian priors for probabilities, A/B testing",
            "formula_latex": r"x^{\alpha-1}(1-x)^{\beta-1}",
        },
    )

    stats["chi_squared_kernel"] = Expression(
        name="chi_squared_kernel",
        symbolic_expr="x**(k/2 - 1) * exp(-x/2)",
        params={"k": 2},
        metadata={
            "category": "statistics",
            "subcategory": "distribution",
            "description": "Chi-squared distribution kernel (unnormalized)",
            "usage": "Hypothesis testing, goodness-of-fit",
            "formula_latex": r"x^{k/2 - 1} e^{-x/2}",
        },
    )

    stats["student_t_kernel"] = Expression(
        name="student_t_kernel",
        symbolic_expr="(1 + x**2 / nu)**(-(nu + 1) / 2)",
        params={"nu": 3},
        metadata={
            "category": "statistics",
            "subcategory": "distribution",
            "description": "Student's t-distribution kernel (unnormalized)",
            "usage": "Small sample inference, robust regression",
            "formula_latex": r"(1 + x^2/\nu)^{-(\nu+1)/2}",
        },
    )

    # ================================================================
    # Cumulative Distribution & Survival Functions
    # ================================================================

    stats["logistic_cdf"] = Expression(
        name="logistic_cdf",
        symbolic_expr="1 / (1 + exp(-(x - mu) / s))",
        params={"mu": 0, "s": 1},
        metadata={
            "category": "statistics",
            "subcategory": "cdf",
            "description": "Logistic CDF (sigmoid with location/scale)",
            "usage": "Binary classification, logistic regression",
            "formula_latex": r"\frac{1}{1 + e^{-(x-\mu)/s}}",
        },
    )

    stats["exponential_cdf"] = Expression(
        name="exponential_cdf",
        symbolic_expr="1 - exp(-lambda_param * x)",
        params={"lambda_param": 1},
        metadata={
            "category": "statistics",
            "subcategory": "cdf",
            "description": "Exponential CDF",
            "usage": "Survival analysis, reliability engineering",
            "formula_latex": r"1 - e^{-\lambda x}",
        },
    )

    stats["survival_exponential"] = Expression(
        name="survival_exponential",
        symbolic_expr="exp(-lambda_param * x)",
        params={"lambda_param": 1},
        metadata={
            "category": "statistics",
            "subcategory": "survival",
            "description": "Exponential survival function S(x) = 1 - F(x)",
            "usage": "Survival analysis, reliability, time-to-event",
            "formula_latex": r"e^{-\lambda x}",
        },
    )

    stats["weibull_survival"] = Expression(
        name="weibull_survival",
        symbolic_expr="exp(-(x / lambda_param)**k)",
        params={"lambda_param": 1, "k": 1.5},
        metadata={
            "category": "statistics",
            "subcategory": "survival",
            "description": "Weibull survival function",
            "usage": "Failure analysis, wind speed distribution",
            "formula_latex": r"e^{-(x/\lambda)^k}",
        },
    )

    # ================================================================
    # Information Theory
    # ================================================================

    stats["binary_entropy"] = Expression(
        name="binary_entropy",
        symbolic_expr="-(p * log(p + 1e-15) + (1 - p) * log(1 - p + 1e-15))",
        metadata={
            "category": "statistics",
            "subcategory": "information_theory",
            "description": "Binary entropy H(p) for Bernoulli variable",
            "usage": "Decision trees, information gain, uncertainty",
            "formula_latex": r"-[p\ln p + (1-p)\ln(1-p)]",
        },
    )

    stats["cross_entropy_elem"] = Expression(
        name="cross_entropy_elem",
        symbolic_expr="-(y * log(p + 1e-15) + (1 - y) * log(1 - p + 1e-15))",
        metadata={
            "category": "statistics",
            "subcategory": "information_theory",
            "description": "Element-wise binary cross-entropy",
            "usage": "Classification loss, KL divergence component",
            "formula_latex": r"-[y\ln p + (1-y)\ln(1-p)]",
        },
    )

    stats["kl_divergence_elem"] = Expression(
        name="kl_divergence_elem",
        symbolic_expr="p * log((p + 1e-15) / (q + 1e-15))",
        metadata={
            "category": "statistics",
            "subcategory": "information_theory",
            "description": "Element-wise KL divergence D_KL(p||q)",
            "usage": "VAE loss, distribution matching, Bayesian inference",
            "formula_latex": r"p \ln\frac{p}{q}",
        },
    )

    stats["js_divergence_elem"] = Expression(
        name="js_divergence_elem",
        symbolic_expr="(1/2) * (p * log((2*p) / (p + q + 1e-15)) + q * log((2*q) / (p + q + 1e-15)))",
        metadata={
            "category": "statistics",
            "subcategory": "information_theory",
            "description": "Element-wise Jensen-Shannon divergence",
            "usage": "GAN training, symmetric distribution comparison",
            "formula_latex": r"\frac{1}{2}[p\ln\frac{2p}{p+q} + q\ln\frac{2q}{p+q}]",
        },
    )

    stats["mutual_info_bound"] = Expression(
        name="mutual_info_bound",
        symbolic_expr="log(1 + exp(x))",
        metadata={
            "category": "statistics",
            "subcategory": "information_theory",
            "description": "Donsker-Varadhan/NWJ lower bound kernel for MI",
            "usage": "Mutual information estimation, representation learning",
            "formula_latex": r"\ln(1 + e^x)",
        },
    )

    # ================================================================
    # Descriptive Statistics (scalar expressions)
    # ================================================================

    stats["z_score"] = Expression(
        name="z_score",
        symbolic_expr="(x - mu) / sigma",
        params={"mu": 0, "sigma": 1},
        metadata={
            "category": "statistics",
            "subcategory": "descriptive",
            "description": "Z-score standardization",
            "usage": "Normalization, hypothesis testing, outlier detection",
            "formula_latex": r"z = \frac{x - \mu}{\sigma}",
        },
    )

    stats["t_statistic"] = Expression(
        name="t_statistic",
        symbolic_expr="(x - mu) / (s / sqrt(n))",
        params={"mu": 0, "s": 1, "n": 30},
        metadata={
            "category": "statistics",
            "subcategory": "descriptive",
            "description": "t-statistic for one-sample t-test",
            "usage": "Hypothesis testing, confidence intervals",
            "formula_latex": r"t = \frac{\bar{x} - \mu}{s/\sqrt{n}}",
        },
    )

    stats["coefficient_of_variation"] = Expression(
        name="coefficient_of_variation",
        symbolic_expr="sigma / mu",
        params={"mu": 1, "sigma": 0.1},
        metadata={
            "category": "statistics",
            "subcategory": "descriptive",
            "description": "Coefficient of variation (relative std dev)",
            "usage": "Compare variability across scales",
            "formula_latex": r"CV = \frac{\sigma}{\mu}",
        },
    )

    stats["pooled_variance"] = Expression(
        name="pooled_variance",
        symbolic_expr="((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2)",
        params={"n1": 30, "s1": 1, "n2": 30, "s2": 1},
        metadata={
            "category": "statistics",
            "subcategory": "descriptive",
            "description": "Pooled variance for two-sample t-test",
            "usage": "Comparing two groups, ANOVA",
            "formula_latex": r"s_p^2 = \frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1+n_2-2}",
        },
    )

    stats["standard_error"] = Expression(
        name="standard_error",
        symbolic_expr="sigma / sqrt(n)",
        params={"sigma": 1, "n": 30},
        metadata={
            "category": "statistics",
            "subcategory": "descriptive",
            "description": "Standard error of the mean",
            "usage": "Confidence intervals, significance testing",
            "formula_latex": r"SE = \frac{\sigma}{\sqrt{n}}",
        },
    )

    # ================================================================
    # Bayesian Statistics
    # ================================================================

    stats["bayes_posterior_kernel"] = Expression(
        name="bayes_posterior_kernel",
        symbolic_expr="exp(-((x - mu_prior)**2) / (2 * sigma_prior**2)) * exp(-((y - x)**2) / (2 * sigma_lik**2))",
        params={"mu_prior": 0, "sigma_prior": 1, "sigma_lik": 1},
        metadata={
            "category": "statistics",
            "subcategory": "bayesian",
            "description": "Gaussian prior × Gaussian likelihood (posterior kernel)",
            "usage": "Bayesian inference, conjugate priors",
            "formula_latex": r"\mathcal{N}(x|\mu_0,\sigma_0^2) \cdot \mathcal{N}(y|x,\sigma_l^2)",
        },
    )

    stats["log_prior_normal"] = Expression(
        name="log_prior_normal",
        symbolic_expr="-(x - mu)**2 / (2 * sigma**2) - log(sigma) - log(2 * pi) / 2",
        params={"mu": 0, "sigma": 1},
        metadata={
            "category": "statistics",
            "subcategory": "bayesian",
            "description": "Log of normal prior density",
            "usage": "Log-space Bayesian computation, weight priors in BNNs",
            "formula_latex": r"-\frac{(x-\mu)^2}{2\sigma^2} - \ln\sigma - \frac{\ln 2\pi}{2}",
        },
    )

    stats["evidence_lower_bound"] = Expression(
        name="evidence_lower_bound",
        symbolic_expr="log_likelihood - kl_term",
        params={"log_likelihood": 0, "kl_term": 0},
        metadata={
            "category": "statistics",
            "subcategory": "bayesian",
            "description": "Evidence Lower Bound (ELBO) = E[log p(x|z)] - KL(q||p)",
            "usage": "Variational autoencoders, variational inference",
            "formula_latex": r"ELBO = \mathbb{E}[\log p(x|z)] - D_{KL}(q(z|x) \| p(z))",
        },
    )

    # ================================================================
    # Regression & Correlation (scalar forms)
    # ================================================================

    stats["pearson_r_component"] = Expression(
        name="pearson_r_component",
        symbolic_expr="(x - mu_x) * (y - mu_y) / (sigma_x * sigma_y)",
        params={"mu_x": 0, "mu_y": 0, "sigma_x": 1, "sigma_y": 1},
        metadata={
            "category": "statistics",
            "subcategory": "correlation",
            "description": "Element-wise Pearson correlation component",
            "usage": "Correlation analysis, feature selection",
            "formula_latex": r"\frac{(x-\mu_x)(y-\mu_y)}{\sigma_x \sigma_y}",
        },
    )

    stats["linear_regression_pred"] = Expression(
        name="linear_regression_pred",
        symbolic_expr="beta_0 + beta_1 * x",
        params={"beta_0": 0, "beta_1": 1},
        trainable_params=["beta_0", "beta_1"],
        metadata={
            "category": "statistics",
            "subcategory": "regression",
            "description": "Simple linear regression prediction",
            "usage": "Regression, trend estimation",
            "formula_latex": r"\hat{y} = \beta_0 + \beta_1 x",
        },
    )

    stats["logistic_regression_pred"] = Expression(
        name="logistic_regression_pred",
        symbolic_expr="1 / (1 + exp(-(beta_0 + beta_1 * x)))",
        params={"beta_0": 0, "beta_1": 1},
        trainable_params=["beta_0", "beta_1"],
        metadata={
            "category": "statistics",
            "subcategory": "regression",
            "description": "Logistic regression prediction (binary)",
            "usage": "Binary classification, probability estimation",
            "formula_latex": r"\sigma(\beta_0 + \beta_1 x) = \frac{1}{1+e^{-(\beta_0+\beta_1 x)}}",
        },
    )

    # ================================================================
    # Moment Generating & Characteristic Functions
    # ================================================================

    stats["mgf_normal"] = Expression(
        name="mgf_normal",
        symbolic_expr="exp(mu * t + sigma**2 * t**2 / 2)",
        params={"mu": 0, "sigma": 1},
        metadata={
            "category": "statistics",
            "subcategory": "generating_function",
            "description": "Moment generating function of Normal distribution",
            "usage": "Deriving moments, proving CLT",
            "formula_latex": r"M(t) = e^{\mu t + \sigma^2 t^2/2}",
        },
    )

    stats["mgf_exponential"] = Expression(
        name="mgf_exponential",
        symbolic_expr="lambda_param / (lambda_param - t)",
        params={"lambda_param": 1},
        metadata={
            "category": "statistics",
            "subcategory": "generating_function",
            "description": "Moment generating function of Exponential distribution",
            "usage": "Computing exponential moments",
            "formula_latex": r"M(t) = \frac{\lambda}{\lambda - t}",
        },
    )

    stats["characteristic_normal"] = Expression(
        name="characteristic_normal",
        symbolic_expr="exp(I * mu * t - sigma**2 * t**2 / 2)",
        params={"mu": 0, "sigma": 1},
        metadata={
            "category": "statistics",
            "subcategory": "generating_function",
            "description": "Characteristic function of Normal distribution",
            "usage": "Fourier analysis of distributions, CLT proofs",
            "formula_latex": r"\varphi(t) = e^{i\mu t - \sigma^2 t^2/2}",
        },
    )

    return stats
