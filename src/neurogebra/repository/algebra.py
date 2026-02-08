"""
Algebra expression repository.

Pre-built algebraic expressions commonly used in mathematics, ML, and
scientific computing. Covers polynomials, transcendental functions,
special curves, and fitting functions.
"""

from neurogebra.core.expression import Expression
from typing import Dict


def get_algebra_expressions() -> Dict[str, Expression]:
    """
    Get dictionary of algebraic expressions.

    Returns:
        Dictionary mapping names to Expression instances
    """
    algebra: Dict[str, Expression] = {}

    # ================================================================
    # Polynomial Functions
    # ================================================================

    algebra["linear_eq"] = Expression(
        name="linear_eq",
        symbolic_expr="m*x + b",
        params={"m": 1, "b": 0},
        trainable_params=["m", "b"],
        metadata={
            "category": "algebra",
            "description": "Linear equation: y = mx + b",
            "usage": "Linear regression, line fitting",
        },
    )

    algebra["quadratic"] = Expression(
        name="quadratic",
        symbolic_expr="a*x**2 + b*x + c",
        params={"a": 1, "b": 0, "c": 0},
        trainable_params=["a", "b", "c"],
        metadata={
            "category": "algebra",
            "description": "Quadratic polynomial: ax² + bx + c",
            "usage": "Curve fitting, parabolic approximation",
        },
    )

    algebra["cubic"] = Expression(
        name="cubic",
        symbolic_expr="a*x**3 + b*x**2 + c*x + d",
        params={"a": 1, "b": 0, "c": 0, "d": 0},
        trainable_params=["a", "b", "c", "d"],
        metadata={
            "category": "algebra",
            "description": "Cubic polynomial: ax³ + bx² + cx + d",
            "usage": "Curve fitting, interpolation",
        },
    )

    algebra["quartic"] = Expression(
        name="quartic",
        symbolic_expr="a*x**4 + b*x**3 + c*x**2 + d*x + e",
        params={"a": 1, "b": 0, "c": 0, "d": 0, "e": 0},
        trainable_params=["a", "b", "c", "d", "e"],
        metadata={
            "category": "algebra",
            "description": "Quartic polynomial: ax⁴ + bx³ + cx² + dx + e",
            "usage": "Higher-order curve fitting",
        },
    )

    algebra["monomial"] = Expression(
        name="monomial",
        symbolic_expr="a * x**n",
        params={"a": 1.0, "n": 2.0},
        trainable_params=["a", "n"],
        metadata={
            "category": "algebra",
            "description": "Monomial: a·xⁿ, single-term power expression",
            "usage": "Basis for polynomial approximation",
        },
    )

    # ================================================================
    # Power & Root Functions
    # ================================================================

    algebra["power_law"] = Expression(
        name="power_law",
        symbolic_expr="a * x**n",
        params={"a": 1.0, "n": 2.0},
        trainable_params=["a", "n"],
        metadata={
            "category": "algebra",
            "description": "Power law: a·xⁿ",
            "usage": "Scaling laws, Zipf's law, allometry",
        },
    )

    algebra["inverse_square"] = Expression(
        name="inverse_square",
        symbolic_expr="a / (x**2 + epsilon)",
        params={"a": 1.0, "epsilon": 1e-8},
        metadata={
            "category": "algebra",
            "description": "Inverse square law: a/x²",
            "usage": "Gravity, electric fields, light intensity",
        },
    )

    algebra["square_root"] = Expression(
        name="square_root",
        symbolic_expr="a * sqrt(Abs(x) + epsilon)",
        params={"a": 1.0, "epsilon": 1e-8},
        metadata={
            "category": "algebra",
            "description": "Square root function: a·√|x|",
            "usage": "Sublinear growth, diminishing returns",
        },
    )

    algebra["nth_root"] = Expression(
        name="nth_root",
        symbolic_expr="(Abs(x) + epsilon)**(1/n)",
        params={"n": 3.0, "epsilon": 1e-8},
        metadata={
            "category": "algebra",
            "description": "Nth root: x^(1/n)",
            "usage": "Root extraction, power normalization",
        },
    )

    # ================================================================
    # Exponential & Logarithmic Functions
    # ================================================================

    algebra["exp_decay"] = Expression(
        name="exp_decay",
        symbolic_expr="A * exp(-k * x)",
        params={"A": 1.0, "k": 1.0},
        trainable_params=["A", "k"],
        metadata={
            "category": "algebra",
            "description": "Exponential decay: A·e^(−kx)",
            "usage": "Radioactive decay, learning rate schedules",
        },
    )

    algebra["exp_growth"] = Expression(
        name="exp_growth",
        symbolic_expr="A * exp(k * x)",
        params={"A": 1.0, "k": 0.1},
        trainable_params=["A", "k"],
        metadata={
            "category": "algebra",
            "description": "Exponential growth: A·e^(kx)",
            "usage": "Population growth, compound interest",
        },
    )

    algebra["double_exponential"] = Expression(
        name="double_exponential",
        symbolic_expr="A * exp(-Abs(x - mu) / b)",
        params={"A": 1.0, "mu": 0.0, "b": 1.0},
        trainable_params=["A", "mu", "b"],
        metadata={
            "category": "algebra",
            "description": "Laplace distribution / double exponential",
            "usage": "Robust statistics, L1 regularization prior",
        },
    )

    algebra["logarithmic"] = Expression(
        name="logarithmic",
        symbolic_expr="a * log(x + epsilon) + b",
        params={"a": 1.0, "b": 0.0, "epsilon": 1e-8},
        trainable_params=["a", "b"],
        metadata={
            "category": "algebra",
            "description": "Logarithmic function: a·ln(x) + b",
            "usage": "Diminishing returns, Weber-Fechner law",
        },
    )

    algebra["log_linear"] = Expression(
        name="log_linear",
        symbolic_expr="a * log(1 + exp(b * x))",
        params={"a": 1.0, "b": 1.0},
        trainable_params=["a", "b"],
        metadata={
            "category": "algebra",
            "description": "Log-linear (softplus family): a·log(1+e^(bx))",
            "usage": "Smooth ramp functions, positive outputs",
        },
    )

    # ================================================================
    # Sigmoid & Growth Curves
    # ================================================================

    algebra["logistic"] = Expression(
        name="logistic",
        symbolic_expr="L / (1 + exp(-k * (x - x0)))",
        params={"L": 1.0, "k": 1.0, "x0": 0.0},
        trainable_params=["L", "k", "x0"],
        metadata={
            "category": "algebra",
            "description": "Logistic growth curve (S-curve)",
            "usage": "Population growth, adoption curves",
        },
    )

    algebra["gompertz"] = Expression(
        name="gompertz",
        symbolic_expr="a * exp(-b * exp(-c * x))",
        params={"a": 1.0, "b": 1.0, "c": 1.0},
        trainable_params=["a", "b", "c"],
        metadata={
            "category": "algebra",
            "description": "Gompertz curve - asymmetric sigmoid growth",
            "usage": "Tumor growth, product adoption, mortality",
        },
    )

    algebra["richards_curve"] = Expression(
        name="richards_curve",
        symbolic_expr="K / (1 + exp(-r * (x - x0)))**(1/nu)",
        params={"K": 1.0, "r": 1.0, "x0": 0.0, "nu": 1.0},
        trainable_params=["K", "r", "x0", "nu"],
        metadata={
            "category": "algebra",
            "description": "Richards / generalized logistic curve",
            "usage": "Flexible growth modeling, epidemiology",
        },
    )

    algebra["probit"] = Expression(
        name="probit",
        symbolic_expr="0.5 * (1 + erf(x / sqrt(2)))",
        metadata={
            "category": "algebra",
            "description": "Probit function (Gaussian CDF Φ(x))",
            "usage": "Probit regression, dose-response curves",
        },
    )

    algebra["hill_equation"] = Expression(
        name="hill_equation",
        symbolic_expr="V_max * x**n / (K**n + x**n)",
        params={"V_max": 1.0, "K": 0.5, "n": 2.0},
        trainable_params=["V_max", "K", "n"],
        metadata={
            "category": "algebra",
            "description": "Hill equation - cooperative binding sigmoid",
            "usage": "Biochemistry, pharmacology, dose-response",
        },
    )

    algebra["michaelis_menten"] = Expression(
        name="michaelis_menten",
        symbolic_expr="V_max * x / (K_m + x)",
        params={"V_max": 1.0, "K_m": 0.5},
        trainable_params=["V_max", "K_m"],
        metadata={
            "category": "algebra",
            "description": "Michaelis-Menten enzyme kinetics",
            "usage": "Enzyme kinetics, saturation modeling",
        },
    )

    # ================================================================
    # Probability Distribution Functions
    # ================================================================

    algebra["gaussian"] = Expression(
        name="gaussian",
        symbolic_expr="A * exp(-(x - mu)**2 / (2 * sigma**2))",
        params={"A": 1.0, "mu": 0.0, "sigma": 1.0},
        trainable_params=["A", "mu", "sigma"],
        metadata={
            "category": "algebra",
            "description": "Gaussian (normal) distribution bell curve",
            "usage": "Probability density, Gaussian processes, RBF",
        },
    )

    algebra["cauchy_distribution"] = Expression(
        name="cauchy_distribution",
        symbolic_expr="1 / (pi * gamma_param * (1 + ((x - x0) / gamma_param)**2))",
        params={"x0": 0.0, "gamma_param": 1.0},
        metadata={
            "category": "algebra",
            "description": "Cauchy (Lorentzian) distribution - heavy tails",
            "usage": "Robust statistics, spectral line shapes",
        },
    )

    algebra["student_t"] = Expression(
        name="student_t",
        symbolic_expr=(
            "(1 + x**2/nu)**(-(nu + 1)/2) "
            "* gamma_func((nu + 1)/2) "
            "/ (sqrt(nu * pi) * gamma_func(nu/2))"
        ),
        params={"nu": 5.0},
        metadata={
            "category": "algebra",
            "description": "Student's t-distribution kernel (unnormalized)",
            "usage": "Small sample inference, robust regression",
        },
    )

    algebra["rayleigh"] = Expression(
        name="rayleigh",
        symbolic_expr="(x / sigma**2) * exp(-x**2 / (2 * sigma**2))",
        params={"sigma": 1.0},
        metadata={
            "category": "algebra",
            "description": "Rayleigh distribution density",
            "usage": "Signal processing, wind speed modeling",
        },
    )

    algebra["laplace_distribution"] = Expression(
        name="laplace_distribution",
        symbolic_expr="(1/(2*b)) * exp(-Abs(x - mu)/b)",
        params={"mu": 0.0, "b": 1.0},
        metadata={
            "category": "algebra",
            "description": "Laplace distribution density",
            "usage": "Sparse priors, L1 loss connection, robustness",
        },
    )

    algebra["beta_distribution"] = Expression(
        name="beta_distribution",
        symbolic_expr="x**(alpha - 1) * (1 - x)**(beta_param - 1)",
        params={"alpha": 2.0, "beta_param": 5.0},
        metadata={
            "category": "algebra",
            "description": "Beta distribution kernel (unnormalized)",
            "usage": "Probability on [0,1], Bayesian conjugate prior",
        },
    )

    # ================================================================
    # Trigonometric & Periodic Functions
    # ================================================================

    algebra["sinusoidal"] = Expression(
        name="sinusoidal",
        symbolic_expr="A * sin(omega * x + phi)",
        params={"A": 1.0, "omega": 1.0, "phi": 0.0},
        trainable_params=["A", "omega", "phi"],
        metadata={
            "category": "algebra",
            "description": "Sinusoidal wave: A·sin(ωx + φ)",
            "usage": "Signal processing, periodic data fitting",
        },
    )

    algebra["damped_oscillation"] = Expression(
        name="damped_oscillation",
        symbolic_expr="A * exp(-gamma_d * x) * cos(omega * x + phi)",
        params={"A": 1.0, "gamma_d": 0.1, "omega": 1.0, "phi": 0.0},
        trainable_params=["A", "gamma_d", "omega", "phi"],
        metadata={
            "category": "algebra",
            "description": "Damped oscillation: A·e^(-γx)·cos(ωx+φ)",
            "usage": "Spring-mass systems, RLC circuits, decay",
        },
    )

    algebra["fourier_term"] = Expression(
        name="fourier_term",
        symbolic_expr="a0 + a1*cos(x) + b1*sin(x) + a2*cos(2*x) + b2*sin(2*x)",
        params={"a0": 0.0, "a1": 1.0, "b1": 0.0, "a2": 0.0, "b2": 0.0},
        trainable_params=["a0", "a1", "b1", "a2", "b2"],
        metadata={
            "category": "algebra",
            "description": "Fourier series (2 terms): a₀ + Σ(aₙcos(nx) + bₙsin(nx))",
            "usage": "Periodic function approximation, spectral analysis",
        },
    )

    algebra["sawtooth_approx"] = Expression(
        name="sawtooth_approx",
        symbolic_expr=(
            "0.5 - (1/pi) * (sin(x) + sin(2*x)/2 + sin(3*x)/3)"
        ),
        metadata={
            "category": "algebra",
            "description": "Sawtooth wave (Fourier approx, 3 terms)",
            "usage": "Signal processing, waveform generation",
        },
    )

    algebra["square_wave_approx"] = Expression(
        name="square_wave_approx",
        symbolic_expr=(
            "(4/pi) * (sin(x) + sin(3*x)/3 + sin(5*x)/5)"
        ),
        metadata={
            "category": "algebra",
            "description": "Square wave (Fourier approx, 3 terms)",
            "usage": "Digital signals, pulse train approximation",
        },
    )

    # ================================================================
    # Kernel Functions (ML / Gaussian Processes)
    # ================================================================

    algebra["rbf_kernel"] = Expression(
        name="rbf_kernel",
        symbolic_expr="exp(-gamma_k * (x - y)**2)",
        params={"gamma_k": 1.0},
        metadata={
            "category": "algebra",
            "description": "RBF / Gaussian kernel: exp(−γ(x−y)²)",
            "usage": "SVM, Gaussian processes, kernel methods",
        },
    )

    algebra["polynomial_kernel"] = Expression(
        name="polynomial_kernel",
        symbolic_expr="(alpha_k * x * y + c_k)**d_k",
        params={"alpha_k": 1.0, "c_k": 1.0, "d_k": 3.0},
        metadata={
            "category": "algebra",
            "description": "Polynomial kernel: (αxy + c)^d",
            "usage": "SVM, polynomial feature maps",
        },
    )

    algebra["laplacian_kernel"] = Expression(
        name="laplacian_kernel",
        symbolic_expr="exp(-gamma_k * Abs(x - y))",
        params={"gamma_k": 1.0},
        metadata={
            "category": "algebra",
            "description": "Laplacian kernel: exp(−γ|x−y|)",
            "usage": "SVM, non-smooth data, Gaussian processes",
        },
    )

    algebra["rational_quadratic_kernel"] = Expression(
        name="rational_quadratic_kernel",
        symbolic_expr="(1 + (x - y)**2 / (2*alpha_rq * length_scale**2))**(-alpha_rq)",
        params={"alpha_rq": 1.0, "length_scale": 1.0},
        metadata={
            "category": "algebra",
            "description": "Rational quadratic kernel (infinite RBF mixture)",
            "usage": "Gaussian processes, multi-scale modeling",
        },
    )

    algebra["matern_12_kernel"] = Expression(
        name="matern_12_kernel",
        symbolic_expr="sigma_k**2 * exp(-Abs(x - y) / length_scale)",
        params={"sigma_k": 1.0, "length_scale": 1.0},
        metadata={
            "category": "algebra",
            "description": "Matérn kernel (ν=1/2) - equivalent to Laplacian",
            "usage": "Gaussian processes for rough functions",
        },
    )

    algebra["matern_32_kernel"] = Expression(
        name="matern_32_kernel",
        symbolic_expr=(
            "sigma_k**2 * (1 + sqrt(3)*Abs(x - y)/length_scale) "
            "* exp(-sqrt(3)*Abs(x - y)/length_scale)"
        ),
        params={"sigma_k": 1.0, "length_scale": 1.0},
        metadata={
            "category": "algebra",
            "description": "Matérn kernel (ν=3/2) - once differentiable",
            "usage": "Gaussian processes, moderate smoothness",
        },
    )

    algebra["periodic_kernel"] = Expression(
        name="periodic_kernel",
        symbolic_expr="sigma_k**2 * exp(-2 * sin(pi * Abs(x - y) / period)**2 / length_scale**2)",
        params={"sigma_k": 1.0, "length_scale": 1.0, "period": 1.0},
        metadata={
            "category": "algebra",
            "description": "Periodic kernel for repeating patterns",
            "usage": "Gaussian processes on periodic data",
        },
    )

    # ================================================================
    # Rational Functions
    # ================================================================

    algebra["rational"] = Expression(
        name="rational",
        symbolic_expr="(a*x + b) / (c*x + d + epsilon)",
        params={"a": 1.0, "b": 0.0, "c": 0.0, "d": 1.0, "epsilon": 1e-8},
        trainable_params=["a", "b", "c", "d"],
        metadata={
            "category": "algebra",
            "description": "Rational function (ax+b)/(cx+d)",
            "usage": "Padé approximation, Möbius transforms",
        },
    )

    algebra["lorentzian"] = Expression(
        name="lorentzian",
        symbolic_expr="A / (1 + ((x - x0)/gamma_l)**2)",
        params={"A": 1.0, "x0": 0.0, "gamma_l": 1.0},
        trainable_params=["A", "x0", "gamma_l"],
        metadata={
            "category": "algebra",
            "description": "Lorentzian / Cauchy peak function",
            "usage": "Spectral line fitting, resonance curves",
        },
    )

    # ================================================================
    # Special Functions & Other
    # ================================================================

    algebra["heaviside"] = Expression(
        name="heaviside",
        symbolic_expr="Piecewise((0, x < 0), (1, True))",
        metadata={
            "category": "algebra",
            "description": "Heaviside step function H(x)",
            "usage": "Signal processing, threshold operations",
        },
    )

    algebra["ramp"] = Expression(
        name="ramp",
        symbolic_expr="Max(0, x)",
        metadata={
            "category": "algebra",
            "description": "Ramp function (same as ReLU)",
            "usage": "Piecewise linear modeling, half-wave rectifier",
        },
    )

    algebra["absolute_value"] = Expression(
        name="absolute_value",
        symbolic_expr="Abs(x)",
        metadata={
            "category": "algebra",
            "description": "Absolute value |x|",
            "usage": "Distance, L1 norm, error magnitude",
        },
    )

    algebra["sign_function"] = Expression(
        name="sign_function",
        symbolic_expr="Piecewise((-1, x < 0), (0, Eq(x, 0)), (1, True))",
        metadata={
            "category": "algebra",
            "description": "Sign / signum function: sgn(x)",
            "usage": "Direction indicator, binary quantization",
        },
    )

    algebra["clamp"] = Expression(
        name="clamp",
        symbolic_expr="Max(low, Min(high, x))",
        params={"low": 0.0, "high": 1.0},
        metadata={
            "category": "algebra",
            "description": "Clamp function: clip x to [low, high]",
            "usage": "Gradient clipping, bounded outputs",
        },
    )

    algebra["lerp"] = Expression(
        name="lerp",
        symbolic_expr="a_val + t * (b_val - a_val)",
        params={"a_val": 0.0, "b_val": 1.0},
        metadata={
            "category": "algebra",
            "description": "Linear interpolation: a + t(b − a)",
            "usage": "Animation, EMA, parameter mixing",
        },
    )

    algebra["smoothstep"] = Expression(
        name="smoothstep",
        symbolic_expr="3*t**2 - 2*t**3",
        metadata={
            "category": "algebra",
            "description": "Smoothstep (Hermite interpolation): 3t² − 2t³",
            "usage": "Smooth transitions, animation easing",
        },
    )

    algebra["smootherstep"] = Expression(
        name="smootherstep",
        symbolic_expr="6*t**5 - 15*t**4 + 10*t**3",
        metadata={
            "category": "algebra",
            "description": "Smootherstep (Ken Perlin): 6t⁵ − 15t⁴ + 10t³",
            "usage": "Perlin noise, ultra-smooth transitions",
        },
    )

    return algebra
