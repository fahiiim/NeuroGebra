"""
Activation function repository.

Pre-built activation functions commonly used in neural networks.
"""

from neurogebra.core.expression import Expression
from typing import Dict


def get_activations() -> Dict[str, Expression]:
    """
    Get dictionary of activation function expressions.

    Returns:
        Dictionary mapping names to Expression instances
    """
    acts: Dict[str, Expression] = {}

    # ReLU
    acts["relu"] = Expression(
        name="relu",
        symbolic_expr="Max(0, x)",
        metadata={
            "category": "activation",
            "description": "Rectified Linear Unit - outputs x if x > 0, else 0",
            "usage": "Most common activation for hidden layers",
            "pros": ["Fast computation", "No vanishing gradient for positive values"],
            "cons": ["Dead neurons for negative values", "Not zero-centered"],
        },
    )

    # Sigmoid
    acts["sigmoid"] = Expression(
        name="sigmoid",
        symbolic_expr="1 / (1 + exp(-x))",
        metadata={
            "category": "activation",
            "description": "Sigmoid function - maps input to (0, 1)",
            "usage": "Binary classification output layer",
            "pros": ["Smooth gradient", "Bounded output (0, 1)"],
            "cons": ["Vanishing gradient", "Not zero-centered", "Expensive exp()"],
        },
    )

    # Tanh
    acts["tanh"] = Expression(
        name="tanh",
        symbolic_expr="tanh(x)",
        metadata={
            "category": "activation",
            "description": "Hyperbolic tangent - maps input to (-1, 1)",
            "usage": "Hidden layers when zero-centered outputs preferred",
            "pros": ["Zero-centered", "Smooth gradient"],
            "cons": ["Vanishing gradient for large values"],
        },
    )

    # Leaky ReLU
    acts["leaky_relu"] = Expression(
        name="leaky_relu",
        symbolic_expr="Max(alpha*x, x)",
        params={"alpha": 0.01},
        metadata={
            "category": "activation",
            "description": "ReLU with small slope for negative values",
            "usage": "Alternative to ReLU to prevent dead neurons",
            "pros": ["Prevents dead neurons", "Fast computation"],
            "cons": ["Inconsistent predictions for negative values"],
        },
    )

    # Swish (SiLU)
    acts["swish"] = Expression(
        name="swish",
        symbolic_expr="x / (1 + exp(-x))",
        metadata={
            "category": "activation",
            "description": "Self-gated activation function (x * sigmoid(x))",
            "usage": "Modern deep networks, often outperforms ReLU",
            "pros": ["Smooth", "Non-monotonic", "Self-gating property"],
            "cons": ["More expensive than ReLU"],
        },
    )

    # GELU
    acts["gelu"] = Expression(
        name="gelu",
        symbolic_expr="0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x**3)))",
        metadata={
            "category": "activation",
            "description": "Gaussian Error Linear Unit",
            "usage": "Transformers, BERT, and GPT models",
            "pros": ["Smooth", "Probabilistic interpretation", "State-of-the-art"],
            "cons": ["Expensive computation"],
        },
    )

    # Softplus
    acts["softplus"] = Expression(
        name="softplus",
        symbolic_expr="log(1 + exp(x))",
        metadata={
            "category": "activation",
            "description": "Smooth approximation of ReLU",
            "usage": "When a smooth differentiable version of ReLU is needed",
            "pros": ["Smooth everywhere", "Always positive output"],
            "cons": ["Slower than ReLU"],
        },
    )

    # ELU
    acts["elu"] = Expression(
        name="elu",
        symbolic_expr="Piecewise((x, x > 0), (alpha*(exp(x) - 1), True))",
        params={"alpha": 1.0},
        metadata={
            "category": "activation",
            "description": "Exponential Linear Unit",
            "usage": "When negative outputs improve robustness",
            "pros": ["Zero-centered for negative inputs", "Smooth"],
            "cons": ["Expensive exp() for negative values"],
        },
    )

    # SELU (Scaled ELU)
    acts["selu"] = Expression(
        name="selu",
        symbolic_expr=(
            "1.0507009873554805 * "
            "Piecewise((x, x > 0), "
            "(1.6732632423543772*(exp(x) - 1), True))"
        ),
        metadata={
            "category": "activation",
            "description": "Scaled Exponential Linear Unit - self-normalizing",
            "usage": "Self-normalizing neural networks (SNNs)",
            "pros": ["Self-normalizing", "Avoids vanishing/exploding gradients"],
            "cons": ["Only effective with specific architectures"],
        },
    )

    # Mish
    acts["mish"] = Expression(
        name="mish",
        symbolic_expr="x * tanh(log(1 + exp(x)))",
        metadata={
            "category": "activation",
            "description": "Mish activation - smooth, non-monotonic",
            "usage": "Computer vision models, YOLOv4",
            "pros": ["Smooth", "Non-monotonic", "Unbounded above"],
            "cons": ["Computationally expensive"],
        },
    )

    # Hard Sigmoid
    acts["hard_sigmoid"] = Expression(
        name="hard_sigmoid",
        symbolic_expr="Max(0, Min(1, (x + 3) / 6))",
        metadata={
            "category": "activation",
            "description": "Piecewise linear approximation of sigmoid",
            "usage": "Mobile/embedded models for efficiency",
            "pros": ["Very fast", "Good approximation of sigmoid"],
            "cons": ["Not smooth at transitions"],
        },
    )

    # Hard Swish
    acts["hard_swish"] = Expression(
        name="hard_swish",
        symbolic_expr="x * Max(0, Min(1, (x + 3) / 6))",
        metadata={
            "category": "activation",
            "description": "Efficient approximation of Swish",
            "usage": "MobileNetV3 and efficient mobile architectures",
            "pros": ["Fast", "Good approximation of Swish"],
            "cons": ["Not smooth at transitions"],
        },
    )

    # Softsign
    acts["softsign"] = Expression(
        name="softsign",
        symbolic_expr="x / (1 + Abs(x))",
        metadata={
            "category": "activation",
            "description": "Maps input to (-1, 1), similar to tanh but lighter tails",
            "usage": "Alternative to tanh for lighter-tail distribution",
            "pros": ["Lighter tails than tanh", "Smooth"],
            "cons": ["Slower convergence than tanh"],
        },
    )

    # Identity / Linear
    acts["linear"] = Expression(
        name="linear",
        symbolic_expr="x",
        metadata={
            "category": "activation",
            "description": "Identity/linear activation - no transformation",
            "usage": "Output layer for regression tasks",
            "pros": ["Simple", "No information loss"],
            "cons": ["No nonlinearity"],
        },
    )

    # Square
    acts["square"] = Expression(
        name="square",
        symbolic_expr="x**2",
        metadata={
            "category": "activation",
            "description": "Square activation function",
            "usage": "Polynomial networks, kernel approximation",
            "pros": ["Simple nonlinearity"],
            "cons": ["Unbounded", "Not monotonic"],
        },
    )

    return acts
