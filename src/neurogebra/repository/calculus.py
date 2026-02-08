"""
Calculus expression repository.

Pre-built calculus expressions: elementary functions, trigonometry,
hyperbolic functions, special functions, series approximations,
and fundamental calculus operations.
"""

from neurogebra.core.expression import Expression
from typing import Dict


def get_calculus_expressions() -> Dict[str, Expression]:
    """
    Get dictionary of common calculus expressions.

    Returns:
        Dictionary mapping names to Expression instances
    """
    calc: Dict[str, Expression] = {}

    # ================================================================
    # Elementary Functions
    # ================================================================

    calc["exp"] = Expression(
        name="exp",
        symbolic_expr="exp(x)",
        metadata={
            "category": "calculus",
            "description": "Natural exponential function e^x",
            "usage": "Growth models, probability distributions",
            "derivative": "exp(x)",
            "integral": "exp(x)",
        },
    )

    calc["ln"] = Expression(
        name="ln",
        symbolic_expr="log(x)",
        metadata={
            "category": "calculus",
            "description": "Natural logarithm ln(x)",
            "usage": "Log-scale transformations, information theory",
            "derivative": "1/x",
            "integral": "x*ln(x) - x",
        },
    )

    calc["log2"] = Expression(
        name="log2",
        symbolic_expr="log(x) / log(2)",
        metadata={
            "category": "calculus",
            "description": "Base-2 logarithm log₂(x)",
            "usage": "Information theory, bits, binary entropy",
        },
    )

    calc["log10"] = Expression(
        name="log10",
        symbolic_expr="log(x) / log(10)",
        metadata={
            "category": "calculus",
            "description": "Base-10 (common) logarithm log₁₀(x)",
            "usage": "Decibels, pH scale, order of magnitude",
        },
    )

    calc["reciprocal"] = Expression(
        name="reciprocal",
        symbolic_expr="1 / x",
        metadata={
            "category": "calculus",
            "description": "Reciprocal function 1/x",
            "usage": "Inverse transformations, harmonic series",
            "derivative": "-1/x²",
        },
    )

    calc["sqrt"] = Expression(
        name="sqrt",
        symbolic_expr="sqrt(x)",
        metadata={
            "category": "calculus",
            "description": "Square root function √x",
            "usage": "Normalization, distance computation",
            "derivative": "1/(2√x)",
        },
    )

    calc["cbrt"] = Expression(
        name="cbrt",
        symbolic_expr="x**(Rational(1, 3))",
        metadata={
            "category": "calculus",
            "description": "Cube root function x^(1/3)",
            "usage": "Volume extraction, cubic equations",
        },
    )

    calc["abs_val"] = Expression(
        name="abs_val",
        symbolic_expr="Abs(x)",
        metadata={
            "category": "calculus",
            "description": "Absolute value |x|",
            "usage": "Distance, L1 norm, magnitude",
        },
    )

    # ================================================================
    # Trigonometric Functions
    # ================================================================

    calc["sin"] = Expression(
        name="sin",
        symbolic_expr="sin(x)",
        metadata={
            "category": "calculus",
            "description": "Sine function",
            "usage": "Trigonometric operations, signal processing",
            "derivative": "cos(x)",
            "integral": "-cos(x)",
        },
    )

    calc["cos"] = Expression(
        name="cos",
        symbolic_expr="cos(x)",
        metadata={
            "category": "calculus",
            "description": "Cosine function",
            "usage": "Trigonometric operations, positional encoding",
            "derivative": "-sin(x)",
            "integral": "sin(x)",
        },
    )

    calc["tan"] = Expression(
        name="tan",
        symbolic_expr="tan(x)",
        metadata={
            "category": "calculus",
            "description": "Tangent function tan(x) = sin(x)/cos(x)",
            "usage": "Trigonometry, angle computation",
            "derivative": "sec²(x) = 1 + tan²(x)",
        },
    )

    calc["sec"] = Expression(
        name="sec",
        symbolic_expr="1 / cos(x)",
        metadata={
            "category": "calculus",
            "description": "Secant function sec(x) = 1/cos(x)",
            "usage": "Advanced trigonometry, integration techniques",
        },
    )

    calc["csc"] = Expression(
        name="csc",
        symbolic_expr="1 / sin(x)",
        metadata={
            "category": "calculus",
            "description": "Cosecant function csc(x) = 1/sin(x)",
            "usage": "Advanced trigonometry",
        },
    )

    calc["cot"] = Expression(
        name="cot",
        symbolic_expr="cos(x) / sin(x)",
        metadata={
            "category": "calculus",
            "description": "Cotangent function cot(x) = cos(x)/sin(x)",
            "usage": "Advanced trigonometry",
        },
    )

    # ================================================================
    # Inverse Trigonometric Functions
    # ================================================================

    calc["arcsin"] = Expression(
        name="arcsin",
        symbolic_expr="asin(x)",
        metadata={
            "category": "calculus",
            "description": "Inverse sine (arcsine) function",
            "usage": "Angle recovery, trigonometric inversion",
            "derivative": "1/√(1 − x²)",
        },
    )

    calc["arccos"] = Expression(
        name="arccos",
        symbolic_expr="acos(x)",
        metadata={
            "category": "calculus",
            "description": "Inverse cosine (arccosine) function",
            "usage": "Angle computation from dot products",
            "derivative": "−1/√(1 − x²)",
        },
    )

    calc["arctan"] = Expression(
        name="arctan",
        symbolic_expr="atan(x)",
        metadata={
            "category": "calculus",
            "description": "Inverse tangent (arctangent) function",
            "usage": "Angle computation, smooth saturation",
            "derivative": "1/(1 + x²)",
        },
    )

    calc["arctan2"] = Expression(
        name="arctan2",
        symbolic_expr="atan2(y, x)",
        metadata={
            "category": "calculus",
            "description": "Two-argument arctangent atan2(y, x)",
            "usage": "Full-circle angle computation in [−π, π]",
        },
    )

    # ================================================================
    # Hyperbolic Functions
    # ================================================================

    calc["sinh"] = Expression(
        name="sinh",
        symbolic_expr="sinh(x)",
        metadata={
            "category": "calculus",
            "description": "Hyperbolic sine: (eˣ − e⁻ˣ)/2",
            "usage": "Catenary curves, special relativity",
            "derivative": "cosh(x)",
        },
    )

    calc["cosh"] = Expression(
        name="cosh",
        symbolic_expr="cosh(x)",
        metadata={
            "category": "calculus",
            "description": "Hyperbolic cosine: (eˣ + e⁻ˣ)/2",
            "usage": "Catenary curves, distance in hyperbolic space",
            "derivative": "sinh(x)",
        },
    )

    calc["tanh_func"] = Expression(
        name="tanh_func",
        symbolic_expr="tanh(x)",
        metadata={
            "category": "calculus",
            "description": "Hyperbolic tangent: sinh(x)/cosh(x)",
            "usage": "Neural network activations, bounded output",
            "derivative": "1 − tanh²(x) = sech²(x)",
        },
    )

    calc["sech"] = Expression(
        name="sech",
        symbolic_expr="1 / cosh(x)",
        metadata={
            "category": "calculus",
            "description": "Hyperbolic secant sech(x) = 1/cosh(x)",
            "usage": "Soliton solutions, sech² potential",
        },
    )

    # ================================================================
    # Inverse Hyperbolic Functions
    # ================================================================

    calc["arcsinh"] = Expression(
        name="arcsinh",
        symbolic_expr="asinh(x)",
        metadata={
            "category": "calculus",
            "description": "Inverse hyperbolic sine: ln(x + √(x²+1))",
            "usage": "Smooth logarithm-like transform for all x",
            "derivative": "1/√(x² + 1)",
        },
    )

    calc["arccosh"] = Expression(
        name="arccosh",
        symbolic_expr="acosh(x)",
        metadata={
            "category": "calculus",
            "description": "Inverse hyperbolic cosine: ln(x + √(x²−1))",
            "usage": "Hyperbolic geometry, distance metrics",
        },
    )

    calc["arctanh"] = Expression(
        name="arctanh",
        symbolic_expr="atanh(x)",
        metadata={
            "category": "calculus",
            "description": "Inverse hyperbolic tangent: ½ln((1+x)/(1−x))",
            "usage": "Fisher z-transform, correlation analysis",
            "derivative": "1/(1 − x²)",
        },
    )

    # ================================================================
    # Special Functions
    # ================================================================

    calc["erf"] = Expression(
        name="erf",
        symbolic_expr="erf(x)",
        metadata={
            "category": "calculus",
            "description": "Error function erf(x) = (2/√π) ∫₀ˣ e^(−t²) dt",
            "usage": "GELU activation, Gaussian CDF, statistics",
            "derivative": "(2/√π)·e^(−x²)",
        },
    )

    calc["erfc"] = Expression(
        name="erfc",
        symbolic_expr="erfc(x)",
        metadata={
            "category": "calculus",
            "description": "Complementary error function: 1 − erf(x)",
            "usage": "Tail probabilities, Q-function",
        },
    )

    calc["gamma_func"] = Expression(
        name="gamma_func",
        symbolic_expr="gamma(x)",
        metadata={
            "category": "calculus",
            "description": "Gamma function Γ(x) - generalized factorial",
            "usage": "Distributions, combinatorics, Γ(n) = (n−1)!",
        },
    )

    calc["digamma"] = Expression(
        name="digamma",
        symbolic_expr="digamma(x)",
        metadata={
            "category": "calculus",
            "description": "Digamma function ψ(x) = d/dx ln Γ(x)",
            "usage": "Bayesian inference, sufficient statistics",
        },
    )

    calc["beta_func"] = Expression(
        name="beta_func",
        symbolic_expr="gamma(a_param) * gamma(b_param) / gamma(a_param + b_param)",
        params={"a_param": 1.0, "b_param": 1.0},
        metadata={
            "category": "calculus",
            "description": "Beta function B(a,b) = Γ(a)Γ(b)/Γ(a+b)",
            "usage": "Beta distribution normalization, Bayesian priors",
        },
    )

    calc["sinc"] = Expression(
        name="sinc",
        symbolic_expr="Piecewise((1, Eq(x, 0)), (sin(pi*x)/(pi*x), True))",
        metadata={
            "category": "calculus",
            "description": "Sinc function: sin(πx)/(πx)",
            "usage": "Signal reconstruction, Fourier analysis",
        },
    )

    # ================================================================
    # Series Approximations (educational)
    # ================================================================

    calc["taylor_exp"] = Expression(
        name="taylor_exp",
        symbolic_expr="1 + x + x**2/2 + x**3/6 + x**4/24 + x**5/120",
        metadata={
            "category": "calculus",
            "description": "Taylor series for eˣ (5 terms)",
            "usage": "Educational: showing how series approximate functions",
        },
    )

    calc["taylor_sin"] = Expression(
        name="taylor_sin",
        symbolic_expr="x - x**3/6 + x**5/120 - x**7/5040",
        metadata={
            "category": "calculus",
            "description": "Taylor series for sin(x) (4 terms)",
            "usage": "Educational: polynomial approximation of sin",
        },
    )

    calc["taylor_cos"] = Expression(
        name="taylor_cos",
        symbolic_expr="1 - x**2/2 + x**4/24 - x**6/720",
        metadata={
            "category": "calculus",
            "description": "Taylor series for cos(x) (4 terms)",
            "usage": "Educational: polynomial approximation of cos",
        },
    )

    calc["taylor_ln1px"] = Expression(
        name="taylor_ln1px",
        symbolic_expr="x - x**2/2 + x**3/3 - x**4/4 + x**5/5",
        metadata={
            "category": "calculus",
            "description": "Taylor series for ln(1+x) (5 terms)",
            "usage": "Educational: approximation valid for |x| < 1",
        },
    )

    calc["taylor_arctan"] = Expression(
        name="taylor_arctan",
        symbolic_expr="x - x**3/3 + x**5/5 - x**7/7",
        metadata={
            "category": "calculus",
            "description": "Taylor series for atan(x) (4 terms)",
            "usage": "Educational: Leibniz formula for π/4",
        },
    )

    # ================================================================
    # Fundamental Calculus Operations
    # ================================================================

    calc["gradient_descent_step"] = Expression(
        name="gradient_descent_step",
        symbolic_expr="w - lr * gradient",
        params={"lr": 0.01},
        metadata={
            "category": "calculus",
            "description": "Single gradient descent update: w ← w − η·∇L",
            "usage": "Foundation of all neural network training",
        },
    )

    calc["chain_rule"] = Expression(
        name="chain_rule",
        symbolic_expr="df_du * du_dx",
        metadata={
            "category": "calculus",
            "description": "Chain rule: d/dx f(u(x)) = f'(u)·u'(x)",
            "usage": "Backpropagation, composite function derivatives",
        },
    )

    calc["product_rule"] = Expression(
        name="product_rule",
        symbolic_expr="f_val * dg_dx + g_val * df_dx",
        metadata={
            "category": "calculus",
            "description": "Product rule: (fg)' = f·g' + g·f'",
            "usage": "Derivatives of products, attention gradients",
        },
    )

    calc["quotient_rule"] = Expression(
        name="quotient_rule",
        symbolic_expr="(g_val * df_dx - f_val * dg_dx) / (g_val**2 + epsilon)",
        params={"epsilon": 1e-8},
        metadata={
            "category": "calculus",
            "description": "Quotient rule: (f/g)' = (g·f' − f·g')/g²",
            "usage": "Derivatives of ratios, softmax gradients",
        },
    )

    calc["finite_difference"] = Expression(
        name="finite_difference",
        symbolic_expr="(f_plus - f_minus) / (2 * h)",
        params={"h": 1e-5},
        metadata={
            "category": "calculus",
            "description": "Central finite difference: (f(x+h) − f(x−h))/(2h)",
            "usage": "Numerical differentiation, gradient checking",
        },
    )

    calc["trapezoidal_rule"] = Expression(
        name="trapezoidal_rule",
        symbolic_expr="h * (f_a + f_b) / 2",
        metadata={
            "category": "calculus",
            "description": "Trapezoidal rule: h·(f(a) + f(b))/2",
            "usage": "Numerical integration, area approximation",
        },
    )

    calc["simpsons_rule"] = Expression(
        name="simpsons_rule",
        symbolic_expr="h * (f_a + 4*f_m + f_b) / 6",
        metadata={
            "category": "calculus",
            "description": "Simpson's rule: h·(f(a) + 4f(m) + f(b))/6",
            "usage": "Accurate numerical integration",
        },
    )

    # ================================================================
    # Integral Transforms (simplified scalar forms)
    # ================================================================

    calc["laplace_kernel"] = Expression(
        name="laplace_kernel",
        symbolic_expr="exp(-s * t)",
        metadata={
            "category": "calculus",
            "description": "Laplace transform kernel e^(−st)",
            "usage": "Control theory, differential equation solutions",
        },
    )

    calc["fourier_kernel_real"] = Expression(
        name="fourier_kernel_real",
        symbolic_expr="cos(omega * t)",
        metadata={
            "category": "calculus",
            "description": "Real part of Fourier kernel cos(ωt)",
            "usage": "Frequency analysis, spectral decomposition",
        },
    )

    calc["fourier_kernel_imag"] = Expression(
        name="fourier_kernel_imag",
        symbolic_expr="-sin(omega * t)",
        metadata={
            "category": "calculus",
            "description": "Imaginary part of Fourier kernel −sin(ωt)",
            "usage": "Frequency analysis, phase information",
        },
    )

    calc["wavelet_morlet"] = Expression(
        name="wavelet_morlet",
        symbolic_expr="exp(-x**2/2) * cos(5*x)",
        metadata={
            "category": "calculus",
            "description": "Morlet wavelet: e^(−x²/2)·cos(5x)",
            "usage": "Time-frequency analysis, wavelet transforms",
        },
    )

    calc["wavelet_mexican_hat"] = Expression(
        name="wavelet_mexican_hat",
        symbolic_expr="(2/sqrt(3)) * pi**(-Rational(1,4)) * (1 - x**2) * exp(-x**2/2)",
        metadata={
            "category": "calculus",
            "description": "Mexican hat (Ricker) wavelet: (1−x²)e^(−x²/2)",
            "usage": "Edge detection, seismology, wavelet analysis",
        },
    )

    return calc
