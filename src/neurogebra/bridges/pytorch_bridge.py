"""
PyTorch bridge for Neurogebra expressions.

Converts Neurogebra expressions to PyTorch-compatible functions.
"""

from typing import Optional
import numpy as np

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from neurogebra.core.expression import Expression


def check_torch():
    """Raise if PyTorch is not installed."""
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for this feature. "
            "Install it with: pip install neurogebra[frameworks] "
            "or pip install torch"
        )


def to_pytorch(expression: Expression) -> "nn.Module":
    """
    Convert a Neurogebra Expression to a PyTorch nn.Module.

    Args:
        expression: Neurogebra Expression instance

    Returns:
        PyTorch nn.Module that implements the expression

    Raises:
        ImportError: If PyTorch is not installed
    """
    check_torch()

    class ExpressionModule(nn.Module):
        def __init__(self, expr: Expression):
            super().__init__()
            self.expr_name = expr.name
            self._sympy_expr = expr.symbolic_expr

            # Create trainable parameters
            for param_name in expr.trainable_params:
                if param_name in expr.params:
                    value = float(expr.params[param_name])
                    setattr(
                        self,
                        param_name,
                        nn.Parameter(torch.tensor(value)),
                    )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # Use numpy for evaluation, convert back
            x_np = x.detach().cpu().numpy()
            result = np.vectorize(
                lambda val: float(
                    self._sympy_expr.subs("x", val)
                )
            )(x_np)
            return torch.tensor(result, dtype=x.dtype, device=x.device)

    return ExpressionModule(expression)


def from_pytorch(module: "nn.Module", name: str = "pytorch_expr") -> Expression:
    """
    Create a Neurogebra Expression from a simple PyTorch activation.

    Args:
        module: PyTorch module (e.g., nn.ReLU())
        name: Name for the expression

    Returns:
        Neurogebra Expression

    Note:
        This creates a numerical-only expression (no symbolic form).
    """
    check_torch()

    # Map known PyTorch modules to symbolic expressions
    module_map = {
        "ReLU": "Max(0, x)",
        "Sigmoid": "1 / (1 + exp(-x))",
        "Tanh": "tanh(x)",
        "Softplus": "log(1 + exp(x))",
        "ELU": "Piecewise((x, x > 0), (exp(x) - 1, True))",
    }

    class_name = module.__class__.__name__

    if class_name in module_map:
        return Expression(
            name=name,
            symbolic_expr=module_map[class_name],
            metadata={"source": "pytorch", "original_class": class_name},
        )

    # Fallback: create expression from numerical evaluation
    return Expression(
        name=name,
        symbolic_expr="x",  # Placeholder
        metadata={
            "source": "pytorch",
            "original_class": class_name,
            "warning": "Symbolic form not available, using placeholder",
        },
    )
