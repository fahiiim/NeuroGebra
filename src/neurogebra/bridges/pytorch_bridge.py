"""
PyTorch bridge for Neurogebra expressions.

Converts Neurogebra expressions to PyTorch-compatible modules.
"""

from typing import Any

import numpy as np
import sympy as sp

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


def model_to_pytorch(model) -> "nn.Module":
    """Convert a Neurogebra Model to a PyTorch nn.Sequential module."""
    check_torch()
    modules = []
    for layer in model.layers:
        ltype = layer.layer_type.lower()
        if ltype == "dense":
            in_dim = (
                layer.weights.shape[0]
                if layer.weights is not None
                else (
                    layer.params.get("input_shape", (1,))[0]
                    if "input_shape" in layer.params
                    else 1
                )
            )
            out_dim = layer.units or 1
            linear = nn.Linear(in_dim, out_dim)
            if layer.weights is not None:
                linear.weight.data = torch.from_numpy(layer.weights.T).float()
            if layer.bias is not None:
                linear.bias.data = torch.from_numpy(layer.bias).float()
            modules.append(linear)
            act = (layer.activation or "").lower()
            if act == "relu":
                modules.append(nn.ReLU())
            elif act == "sigmoid":
                modules.append(nn.Sigmoid())
            elif act == "tanh":
                modules.append(nn.Tanh())
            elif act == "softmax":
                modules.append(nn.Softmax(dim=-1))
            elif act == "leaky_relu":
                modules.append(nn.LeakyReLU(0.01))
            elif act == "gelu":
                modules.append(nn.GELU())
        elif ltype == "dropout":
            modules.append(nn.Dropout(p=layer.params.get("rate", 0.2)))
        elif ltype == "flatten":
            modules.append(nn.Flatten())
        elif ltype == "batchnorm":
            modules.append(nn.BatchNorm1d(layer.params.get("num_features", 1)))
        elif ltype == "conv2d":
            k = layer.params.get("kernel_size", 3)
            c_in = layer.weights.shape[2] if layer.weights is not None else 1
            c_out = layer.units or 32
            padding = (
                k // 2
                if str(layer.params.get("padding", "valid")).lower() == "same"
                else 0
            )
            conv = nn.Conv2d(c_in, c_out, kernel_size=k, padding=padding)
            if layer.weights is not None:
                w_pt = np.transpose(layer.weights, (3, 2, 0, 1))
                conv.weight.data = torch.from_numpy(w_pt).float()
            if layer.bias is not None:
                conv.bias.data = torch.from_numpy(layer.bias).float()
            modules.append(conv)
            act = (layer.activation or "").lower()
            if act == "relu":
                modules.append(nn.ReLU())
            elif act == "sigmoid":
                modules.append(nn.Sigmoid())
        elif ltype == "maxpool2d":
            p = layer.params.get("pool_size", 2)
            modules.append(nn.MaxPool2d(kernel_size=p))

    seq = nn.Sequential(*modules)
    seq.eval()
    return seq


def to_pytorch(obj: Any) -> "nn.Module":
    """
    Convert a Neurogebra Expression or Model to a PyTorch nn.Module.

    Args:
        obj: Neurogebra Expression or Model instance

    Returns:
        PyTorch nn.Module that implements the computation and gradient flow.
    """
    check_torch()
    if hasattr(obj, "layers"):
        return model_to_pytorch(obj)

    expression = obj

    class ExpressionModule(nn.Module):
        def __init__(self, expr: Expression):
            super().__init__()
            self.expr_name = expr.name
            self._symbolic_expr = expr.symbolic_expr
            self._base_params = expr.params.copy()

            free_symbol_map = {
                symbol.name: symbol for symbol in self._symbolic_expr.free_symbols
            }
            param_names = set(self._base_params.keys())

            unresolved_inputs = sorted(
                name for name in free_symbol_map if name not in param_names
            )
            if len(unresolved_inputs) > 1:
                raise ValueError(
                    "to_pytorch currently supports single-input expressions. "
                    f"Found unresolved variables: {unresolved_inputs}"
                )
            self._input_symbol_name = unresolved_inputs[0] if unresolved_inputs else "x"

            self._ordered_symbol_names = sorted(free_symbol_map.keys())
            self._ordered_symbols = [
                free_symbol_map[name] for name in self._ordered_symbol_names
            ]

            # Register trainable scalar parameters on the module.
            self._trainable_names = []
            for param_name in expr.trainable_params:
                if param_name in self._base_params:
                    value = float(self._base_params[param_name])
                else:
                    value = 0.0
                    self._base_params[param_name] = value

                self._trainable_names.append(param_name)
                setattr(
                    self,
                    param_name,
                    nn.Parameter(torch.tensor(value, dtype=torch.float32)),
                )

            self._value_fn = sp.lambdify(
                self._ordered_symbols,
                self._symbolic_expr,
                modules=["numpy"],
            )

            x_symbol = free_symbol_map.get(
                self._input_symbol_name, sp.Symbol(self._input_symbol_name)
            )
            grad_x_expr = sp.diff(self._symbolic_expr, x_symbol)
            self._grad_x_fn = sp.lambdify(
                self._ordered_symbols,
                grad_x_expr,
                modules=["numpy"],
            )

            self._grad_param_fns = {}
            for param_name in self._trainable_names:
                param_symbol = free_symbol_map.get(param_name, sp.Symbol(param_name))
                grad_param_expr = sp.diff(self._symbolic_expr, param_symbol)
                self._grad_param_fns[param_name] = sp.lambdify(
                    self._ordered_symbols,
                    grad_param_expr,
                    modules=["numpy"],
                )

        def _build_symbol_values(self, x_np, param_values):
            symbol_values = dict(self._base_params.items())
            symbol_values.update(param_values)
            symbol_values[self._input_symbol_name] = x_np
            return symbol_values

        def _evaluate_callable(self, fn, ordered_args, target_shape):
            if self._ordered_symbol_names:
                value = fn(*ordered_args)
            else:
                value = fn()

            value_np = np.asarray(value, dtype=np.float64)
            if value_np.shape == ():
                return np.full(target_shape, float(value_np), dtype=np.float64)
            if value_np.shape != target_shape:
                value_np = np.broadcast_to(value_np, target_shape)
            return np.asarray(value_np, dtype=np.float64)

        def _evaluate_with_gradients(self, x: "torch.Tensor", params):
            x_np = x.detach().cpu().numpy()

            param_values = {
                name: float(param.detach().cpu().item())
                for name, param in zip(self._trainable_names, params)
            }

            symbol_values = self._build_symbol_values(x_np, param_values)
            ordered_args = [
                symbol_values.get(name, 0.0) for name in self._ordered_symbol_names
            ]

            output_np = self._evaluate_callable(
                self._value_fn, ordered_args, x_np.shape
            )
            grad_x_np = self._evaluate_callable(
                self._grad_x_fn, ordered_args, x_np.shape
            )
            grad_param_nps = [
                self._evaluate_callable(
                    self._grad_param_fns[name], ordered_args, x_np.shape
                )
                for name in self._trainable_names
            ]

            return output_np, grad_x_np, grad_param_nps

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            module = self
            trainable_params = [getattr(self, name) for name in self._trainable_names]

            class SympyExpressionFunction(torch.autograd.Function):
                @staticmethod
                def forward(ctx, input_tensor, *params):
                    output_np, grad_x_np, grad_param_nps = (
                        module._evaluate_with_gradients(input_tensor, params)
                    )

                    grad_x_tensor = torch.tensor(
                        grad_x_np,
                        dtype=input_tensor.dtype,
                        device=input_tensor.device,
                    )
                    grad_param_tensors = [
                        torch.tensor(
                            grad_np,
                            dtype=input_tensor.dtype,
                            device=input_tensor.device,
                        )
                        for grad_np in grad_param_nps
                    ]

                    ctx.param_dtypes = [param.dtype for param in params]
                    ctx.save_for_backward(grad_x_tensor, *grad_param_tensors)

                    return torch.tensor(
                        output_np,
                        dtype=input_tensor.dtype,
                        device=input_tensor.device,
                    )

                @staticmethod
                def backward(ctx, grad_output):
                    saved_tensors = ctx.saved_tensors

                    grad_x = grad_output * saved_tensors[0]

                    grad_params = []
                    for index, grad_param in enumerate(saved_tensors[1:]):
                        grad_value = (grad_output * grad_param).sum()
                        grad_params.append(grad_value.to(ctx.param_dtypes[index]))

                    return (grad_x, *grad_params)

            return SympyExpressionFunction.apply(x, *trainable_params)

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
