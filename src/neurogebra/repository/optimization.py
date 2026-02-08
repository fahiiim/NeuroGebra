"""
Optimization expression repository.

Pre-built optimization expressions: gradient-based optimizer update rules,
learning rate schedules, convergence criteria, and optimization utilities
used in machine learning training loops.
"""

from neurogebra.core.expression import Expression
from typing import Dict


def get_optimization_expressions() -> Dict[str, Expression]:
    """
    Get dictionary of optimization expressions.

    Returns:
        Dictionary mapping names to Expression instances
    """
    opt: Dict[str, Expression] = {}

    # ================================================================
    # Gradient-Based Optimizer Update Rules
    # ================================================================

    opt["sgd_step"] = Expression(
        name="sgd_step",
        symbolic_expr="w - lr * grad",
        params={"lr": 0.01},
        metadata={
            "category": "optimization",
            "subcategory": "optimizer",
            "description": "Stochastic Gradient Descent: w ← w - η∇L",
            "usage": "Basic optimization, baseline training",
            "pros": ["Simple", "Low memory", "Well-understood"],
            "cons": ["Slow convergence", "Sensitive to learning rate"],
            "formula_latex": r"w_{t+1} = w_t - \eta \nabla L",
        },
    )

    opt["momentum_step"] = Expression(
        name="momentum_step",
        symbolic_expr="w - (mu_momentum * v + lr * grad)",
        params={"lr": 0.01, "mu_momentum": 0.9, "v": 0},
        metadata={
            "category": "optimization",
            "subcategory": "optimizer",
            "description": "SGD with Momentum: v ← μv + η∇L, w ← w - v",
            "usage": "Faster convergence, escaping local minima",
            "pros": ["Accelerates in consistent gradient direction"],
            "cons": ["Extra hyperparameter μ"],
            "formula_latex": r"v_{t+1} = \mu v_t + \eta \nabla L;\; w_{t+1} = w_t - v_{t+1}",
        },
    )

    opt["nesterov_step"] = Expression(
        name="nesterov_step",
        symbolic_expr="w - (mu_momentum * v + lr * grad_lookahead)",
        params={"lr": 0.01, "mu_momentum": 0.9, "v": 0, "grad_lookahead": 0},
        metadata={
            "category": "optimization",
            "subcategory": "optimizer",
            "description": "Nesterov Accelerated Gradient",
            "usage": "Improved momentum, better convergence",
            "pros": ["Look-ahead gradient correction", "Faster convergence"],
            "formula_latex": r"v_{t+1} = \mu v_t + \eta \nabla L(w_t - \mu v_t);\; w_{t+1} = w_t - v_{t+1}",
        },
    )

    opt["adagrad_step"] = Expression(
        name="adagrad_step",
        symbolic_expr="w - lr * grad / (sqrt(G + 1e-8))",
        params={"lr": 0.01, "G": 0},
        metadata={
            "category": "optimization",
            "subcategory": "optimizer",
            "description": "AdaGrad: adaptive per-parameter learning rate",
            "usage": "Sparse data, NLP embeddings",
            "pros": ["Adapts to parameter frequency", "Good for sparse gradients"],
            "cons": ["Learning rate monotonically decreases"],
            "formula_latex": r"w_{t+1} = w_t - \frac{\eta}{\sqrt{G_t + \epsilon}} \nabla L",
        },
    )

    opt["rmsprop_step"] = Expression(
        name="rmsprop_step",
        symbolic_expr="w - lr * grad / (sqrt(v_rms + 1e-8))",
        params={"lr": 0.001, "v_rms": 0, "rho": 0.9},
        metadata={
            "category": "optimization",
            "subcategory": "optimizer",
            "description": "RMSProp: running average of squared gradients",
            "usage": "RNNs, non-stationary objectives",
            "pros": ["Fixes AdaGrad decay", "Adapts to recent gradients"],
            "formula_latex": r"v_t = \rho v_{t-1} + (1-\rho)(\nabla L)^2;\; w_{t+1} = w_t - \frac{\eta}{\sqrt{v_t+\epsilon}} \nabla L",
        },
    )

    opt["adam_step"] = Expression(
        name="adam_step",
        symbolic_expr="w - lr * m_hat / (sqrt(v_hat) + 1e-8)",
        params={"lr": 0.001, "m_hat": 0, "v_hat": 0},
        metadata={
            "category": "optimization",
            "subcategory": "optimizer",
            "description": "Adam: adaptive moments (most popular optimizer)",
            "usage": "Default choice for deep learning",
            "pros": ["Adaptive LR", "Momentum", "Bias correction"],
            "formula_latex": r"w_{t+1} = w_t - \frac{\eta \hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}",
        },
    )

    opt["adamw_step"] = Expression(
        name="adamw_step",
        symbolic_expr="w * (1 - lr * lambda_wd) - lr * m_hat / (sqrt(v_hat) + 1e-8)",
        params={"lr": 0.001, "m_hat": 0, "v_hat": 0, "lambda_wd": 0.01},
        metadata={
            "category": "optimization",
            "subcategory": "optimizer",
            "description": "AdamW: Adam with decoupled weight decay",
            "usage": "Transformers, modern deep learning",
            "pros": ["Better generalization than Adam", "Correct weight decay"],
            "formula_latex": r"w_{t+1} = w_t(1-\eta\lambda) - \frac{\eta\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}",
        },
    )

    opt["adam_m_update"] = Expression(
        name="adam_m_update",
        symbolic_expr="beta1 * m + (1 - beta1) * grad",
        params={"beta1": 0.9, "m": 0},
        metadata={
            "category": "optimization",
            "subcategory": "optimizer",
            "description": "Adam first moment (mean) update",
            "usage": "Component of Adam optimizer",
            "formula_latex": r"m_t = \beta_1 m_{t-1} + (1-\beta_1)\nabla L",
        },
    )

    opt["adam_v_update"] = Expression(
        name="adam_v_update",
        symbolic_expr="beta2 * v_adam + (1 - beta2) * grad**2",
        params={"beta2": 0.999, "v_adam": 0},
        metadata={
            "category": "optimization",
            "subcategory": "optimizer",
            "description": "Adam second moment (variance) update",
            "usage": "Component of Adam optimizer",
            "formula_latex": r"v_t = \beta_2 v_{t-1} + (1-\beta_2)(\nabla L)^2",
        },
    )

    opt["adam_bias_correction_m"] = Expression(
        name="adam_bias_correction_m",
        symbolic_expr="m / (1 - beta1**t)",
        params={"beta1": 0.9, "m": 0, "t": 1},
        metadata={
            "category": "optimization",
            "subcategory": "optimizer",
            "description": "Adam first moment bias correction",
            "usage": "Component of Adam optimizer",
            "formula_latex": r"\hat{m}_t = \frac{m_t}{1-\beta_1^t}",
        },
    )

    opt["adam_bias_correction_v"] = Expression(
        name="adam_bias_correction_v",
        symbolic_expr="v_adam / (1 - beta2**t)",
        params={"beta2": 0.999, "v_adam": 0, "t": 1},
        metadata={
            "category": "optimization",
            "subcategory": "optimizer",
            "description": "Adam second moment bias correction",
            "usage": "Component of Adam optimizer",
            "formula_latex": r"\hat{v}_t = \frac{v_t}{1-\beta_2^t}",
        },
    )

    # ================================================================
    # Learning Rate Schedules
    # ================================================================

    opt["constant_lr"] = Expression(
        name="constant_lr",
        symbolic_expr="lr_init",
        params={"lr_init": 0.001},
        metadata={
            "category": "optimization",
            "subcategory": "lr_schedule",
            "description": "Constant learning rate (baseline)",
            "usage": "Simple training, debugging",
            "formula_latex": r"\eta_t = \eta_0",
        },
    )

    opt["step_decay_lr"] = Expression(
        name="step_decay_lr",
        symbolic_expr="lr_init * gamma_decay**floor(epoch / step_size)",
        params={"lr_init": 0.01, "gamma_decay": 0.1, "step_size": 30},
        metadata={
            "category": "optimization",
            "subcategory": "lr_schedule",
            "description": "Step decay: reduce LR by factor every N epochs",
            "usage": "Image classification (ResNet schedule)",
            "formula_latex": r"\eta_t = \eta_0 \cdot \gamma^{\lfloor t/s \rfloor}",
        },
    )

    opt["exponential_decay_lr"] = Expression(
        name="exponential_decay_lr",
        symbolic_expr="lr_init * exp(-k_decay * epoch)",
        params={"lr_init": 0.01, "k_decay": 0.01},
        metadata={
            "category": "optimization",
            "subcategory": "lr_schedule",
            "description": "Exponential decay learning rate",
            "usage": "Smooth LR reduction over training",
            "formula_latex": r"\eta_t = \eta_0 e^{-kt}",
        },
    )

    opt["cosine_annealing_lr"] = Expression(
        name="cosine_annealing_lr",
        symbolic_expr="lr_min + (lr_max - lr_min) * (1 + cos(pi * epoch / T_max)) / 2",
        params={"lr_min": 1e-6, "lr_max": 0.01, "T_max": 100},
        metadata={
            "category": "optimization",
            "subcategory": "lr_schedule",
            "description": "Cosine annealing schedule (warm restarts capable)",
            "usage": "State-of-the-art training, SGDR",
            "formula_latex": r"\eta_t = \eta_{min} + \frac{\eta_{max}-\eta_{min}}{2}(1+\cos\frac{\pi t}{T_{max}})",
        },
    )

    opt["warmup_linear_lr"] = Expression(
        name="warmup_linear_lr",
        symbolic_expr="lr_target * Min(1, epoch / warmup_steps)",
        params={"lr_target": 0.001, "warmup_steps": 1000},
        metadata={
            "category": "optimization",
            "subcategory": "lr_schedule",
            "description": "Linear warmup (ramp up, then constant)",
            "usage": "Transformers, large batch training",
            "formula_latex": r"\eta_t = \eta_{target} \cdot \min(1, t/t_{warmup})",
        },
    )

    opt["polynomial_decay_lr"] = Expression(
        name="polynomial_decay_lr",
        symbolic_expr="(lr_init - lr_end) * (1 - epoch / max_epochs)**power_lr + lr_end",
        params={"lr_init": 0.01, "lr_end": 1e-6, "max_epochs": 100, "power_lr": 1},
        metadata={
            "category": "optimization",
            "subcategory": "lr_schedule",
            "description": "Polynomial decay learning rate",
            "usage": "Object detection, semantic segmentation",
            "formula_latex": r"\eta_t = (\eta_0 - \eta_{end})(1 - t/T)^p + \eta_{end}",
        },
    )

    opt["inverse_sqrt_lr"] = Expression(
        name="inverse_sqrt_lr",
        symbolic_expr="lr_init / sqrt(Max(epoch, warmup_steps))",
        params={"lr_init": 0.01, "warmup_steps": 4000},
        metadata={
            "category": "optimization",
            "subcategory": "lr_schedule",
            "description": "Inverse square root schedule (Transformer original)",
            "usage": "Attention Is All You Need schedule",
            "formula_latex": r"\eta_t = \frac{\eta_0}{\sqrt{\max(t, t_w)}}",
        },
    )

    opt["cyclical_lr"] = Expression(
        name="cyclical_lr",
        symbolic_expr="lr_min + (lr_max - lr_min) * Abs(2 * (epoch / (2 * step_size) - floor(epoch / (2 * step_size) + Rational(1,2))))",
        params={"lr_min": 1e-4, "lr_max": 0.01, "step_size": 2000},
        metadata={
            "category": "optimization",
            "subcategory": "lr_schedule",
            "description": "Cyclical learning rate (triangular policy)",
            "usage": "Super-convergence, finding optimal LR range",
            "formula_latex": r"\eta_t = \eta_{min} + (\eta_{max}-\eta_{min}) \cdot \text{tri}(t)",
        },
    )

    opt["one_cycle_approx_lr"] = Expression(
        name="one_cycle_approx_lr",
        symbolic_expr="lr_max * (1 + cos(pi * epoch / T_max)) / 2",
        params={"lr_max": 0.01, "T_max": 100},
        metadata={
            "category": "optimization",
            "subcategory": "lr_schedule",
            "description": "1-cycle policy approximation (cosine phase)",
            "usage": "Fast training with super-convergence",
            "formula_latex": r"\eta_t \approx \frac{\eta_{max}}{2}(1 + \cos\frac{\pi t}{T})",
        },
    )

    # ================================================================
    # Gradient Clipping & Processing
    # ================================================================

    opt["gradient_clip_value"] = Expression(
        name="gradient_clip_value",
        symbolic_expr="Max(Min(grad, clip_val), -clip_val)",
        params={"clip_val": 1.0},
        metadata={
            "category": "optimization",
            "subcategory": "gradient_processing",
            "description": "Gradient clipping by value",
            "usage": "Prevent exploding gradients in RNNs",
            "formula_latex": r"\text{clip}(g, -c, c)",
        },
    )

    opt["gradient_clip_norm"] = Expression(
        name="gradient_clip_norm",
        symbolic_expr="grad * Min(1, max_norm / (grad_norm + 1e-8))",
        params={"max_norm": 1.0, "grad_norm": 1.0},
        metadata={
            "category": "optimization",
            "subcategory": "gradient_processing",
            "description": "Gradient clipping by global norm",
            "usage": "Standard gradient clipping for deep networks",
            "formula_latex": r"g \cdot \min(1, \frac{c}{\|g\|})",
        },
    )

    opt["ema_update"] = Expression(
        name="ema_update",
        symbolic_expr="alpha_ema * param_new + (1 - alpha_ema) * param_ema",
        params={"alpha_ema": 0.001},
        metadata={
            "category": "optimization",
            "subcategory": "gradient_processing",
            "description": "Exponential Moving Average parameter update",
            "usage": "Model averaging, Polyak averaging, EMA models",
            "formula_latex": r"\bar{\theta}_t = \alpha\theta_t + (1-\alpha)\bar{\theta}_{t-1}",
        },
    )

    # ================================================================
    # Loss Landscape & Convergence
    # ================================================================

    opt["quadratic_bowl"] = Expression(
        name="quadratic_bowl",
        symbolic_expr="a_qb * (x - x_opt)**2 + b_qb * (y - y_opt)**2",
        params={"a_qb": 1, "b_qb": 1, "x_opt": 0, "y_opt": 0},
        metadata={
            "category": "optimization",
            "subcategory": "landscape",
            "description": "Quadratic bowl (simple convex surface)",
            "usage": "Visualizing optimizer trajectories",
            "formula_latex": r"f(x,y) = a(x-x^*)^2 + b(y-y^*)^2",
        },
    )

    opt["rosenbrock"] = Expression(
        name="rosenbrock",
        symbolic_expr="(a_rb - x)**2 + b_rb * (y - x**2)**2",
        params={"a_rb": 1, "b_rb": 100},
        metadata={
            "category": "optimization",
            "subcategory": "landscape",
            "description": "Rosenbrock function (banana function)",
            "usage": "Optimizer benchmarking, non-convex optimization testing",
            "formula_latex": r"f(x,y) = (a-x)^2 + b(y-x^2)^2",
        },
    )

    opt["rastrigin_2d"] = Expression(
        name="rastrigin_2d",
        symbolic_expr="20 + (x**2 - 10*cos(2*pi*x)) + (y**2 - 10*cos(2*pi*y))",
        metadata={
            "category": "optimization",
            "subcategory": "landscape",
            "description": "Rastrigin function (many local minima)",
            "usage": "Global optimization algorithms, swarm intelligence",
            "formula_latex": r"f(\mathbf{x}) = An + \sum[x_i^2 - A\cos(2\pi x_i)]",
        },
    )

    opt["ackley_2d"] = Expression(
        name="ackley_2d",
        symbolic_expr="-20 * exp(-0.2 * sqrt(0.5 * (x**2 + y**2))) - exp(0.5 * (cos(2*pi*x) + cos(2*pi*y))) + exp(1) + 20",
        metadata={
            "category": "optimization",
            "subcategory": "landscape",
            "description": "Ackley function (nearly flat outer region)",
            "usage": "Testing optimizer escape from flat regions",
            "formula_latex": r"-20e^{-0.2\sqrt{0.5(x^2+y^2)}} - e^{0.5(\cos 2\pi x + \cos 2\pi y)} + e + 20",
        },
    )

    return opt
