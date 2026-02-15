"""
FormulaRenderer — Pretty-print mathematical formulas in the terminal.

Converts SymPy expressions to coloured Unicode/ASCII art, LaTeX strings,
and inline formula annotations for forward and backward passes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

try:
    import sympy as sp
    from sympy import Symbol, latex, pretty
    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False


class FormulaRenderer:
    """
    Render mathematical formulas with colour in the terminal.

    Supports:
    - Forward-pass formulas  (e.g. ``σ(Wx + b)``)
    - Backward-pass gradient formulas (e.g. ``∂L/∂W = (∂L/∂y)(xᵀ)``)
    - Full model equation composition across layers
    - LaTeX export for documentation
    """

    # Common operation → Unicode symbol mapping
    _OP_SYMBOLS = {
        "relu": "ReLU(z) = max(0, z)",
        "sigmoid": "σ(z) = 1 / (1 + e⁻ᶻ)",
        "tanh": "tanh(z)",
        "softmax": "softmax(zᵢ) = eᶻⁱ / Σeᶻʲ",
        "linear": "f(z) = z",
        "leaky_relu": "LeakyReLU(z) = max(αz, z)",
        "elu": "ELU(z) = z if z>0 else α(eᶻ-1)",
        "swish": "swish(z) = z·σ(z)",
        "gelu": "GELU(z) ≈ z·Φ(z)",
        "mish": "mish(z) = z·tanh(softplus(z))",
    }

    _GRAD_SYMBOLS = {
        "relu": "∂ReLU/∂z = 1 if z>0 else 0",
        "sigmoid": "∂σ/∂z = σ(z)(1−σ(z))",
        "tanh": "∂tanh/∂z = 1 − tanh²(z)",
        "softmax": "∂softmax/∂z = softmax(z)(δᵢⱼ − softmax(z))",
        "linear": "∂f/∂z = 1",
        "leaky_relu": "∂LeakyReLU/∂z = 1 if z>0 else α",
        "swish": "∂swish/∂z = swish(z) + σ(z)(1 − swish(z))",
    }

    def __init__(self):
        self._console = Console() if HAS_RICH else None

    # ------------------------------------------------------------------
    # Layer formulas
    # ------------------------------------------------------------------

    def dense_forward_formula(self, layer_name: str, activation: str = "linear",
                              input_symbol: str = "x", layer_idx: int = 0) -> str:
        """Return the forward formula string for a Dense layer."""
        act = self._OP_SYMBOLS.get(activation, activation)
        w = f"W{layer_idx}"
        b = f"b{layer_idx}"
        z = f"z{layer_idx}"
        a = f"a{layer_idx}"
        lines = [
            f"  {z} = {w}·{input_symbol} + {b}",
            f"  {a} = {act.split('=')[0].strip()}({z})",
        ]
        return "\n".join(lines)

    def dense_backward_formula(self, layer_name: str, activation: str = "linear",
                               layer_idx: int = 0) -> str:
        """Return the backward formula string for a Dense layer."""
        grad = self._GRAD_SYMBOLS.get(activation, f"∂{activation}/∂z")
        w = f"W{layer_idx}"
        a_prev = f"a{layer_idx - 1}" if layer_idx > 0 else "x"
        lines = [
            f"  ∂L/∂z{layer_idx} = ∂L/∂a{layer_idx} ⊙ {grad}",
            f"  ∂L/∂{w} = ∂L/∂z{layer_idx} · ({a_prev})ᵀ",
            f"  ∂L/∂b{layer_idx} = Σ ∂L/∂z{layer_idx}",
            f"  ∂L/∂{a_prev} = ({w})ᵀ · ∂L/∂z{layer_idx}",
        ]
        return "\n".join(lines)

    def loss_formula(self, loss_name: str) -> str:
        """Return commonly-used loss formula."""
        formulas = {
            "mse": "L = (1/n) Σ(yᵢ − ŷᵢ)²",
            "mae": "L = (1/n) Σ|yᵢ − ŷᵢ|",
            "binary_crossentropy": "L = −(1/n) Σ[yᵢ log(ŷᵢ) + (1−yᵢ) log(1−ŷᵢ)]",
            "cross_entropy": "L = −Σ yᵢ log(ŷᵢ)",
            "huber": "L = { ½δ²  if |y−ŷ|≤δ; δ|y−ŷ|−½δ²  otherwise }",
            "hinge": "L = Σ max(0, 1 − yᵢ·ŷᵢ)",
        }
        return formulas.get(loss_name, f"L = {loss_name}(y, ŷ)")

    def loss_gradient_formula(self, loss_name: str) -> str:
        formulas = {
            "mse": "∂L/∂ŷ = (2/n)(ŷ − y)",
            "mae": "∂L/∂ŷ = (1/n) sign(ŷ − y)",
            "binary_crossentropy": "∂L/∂ŷ = −(y/ŷ) + (1−y)/(1−ŷ)",
            "cross_entropy": "∂L/∂ŷ = −y/ŷ",
        }
        return formulas.get(loss_name, f"∂L/∂ŷ = ∂{loss_name}/∂ŷ")

    # ------------------------------------------------------------------
    # Composition
    # ------------------------------------------------------------------

    def full_model_equation(self, layer_specs: List[Dict[str, Any]]) -> str:
        """Compose the entire model into one chained equation."""
        eq_parts = []
        input_sym = "x"
        for i, spec in enumerate(layer_specs):
            ltype = spec.get("type", "dense")
            act = spec.get("activation", "linear")
            if ltype == "dense":
                act_sym = self._OP_SYMBOLS.get(act, act).split("=")[0].strip()
                eq_parts.append(f"{act_sym}(W{i}·{input_sym} + b{i})")
                input_sym = f"a{i}"
            elif ltype == "dropout":
                eq_parts.append(f"Dropout(p={spec.get('rate', 0.2)})")
            elif ltype == "flatten":
                eq_parts.append("Flatten")
            elif ltype == "batchnorm":
                eq_parts.append("BatchNorm")
        return "ŷ = " + " → ".join(eq_parts) if eq_parts else "ŷ = f(x)"

    # ------------------------------------------------------------------
    # Pretty printing
    # ------------------------------------------------------------------

    def render(self, formula: str, title: str = "Formula",
               style: str = "magenta") -> None:
        """Print a formula panel to the terminal."""
        if self._console:
            self._console.print(Panel(
                f"[{style}]{formula}[/]",
                title=f"[bold {style}]{title}[/]",
                border_style=style,
            ))
        else:
            print(f"\n{title}: {formula}\n")

    def render_forward(self, layer_name: str, activation: str,
                       layer_idx: int) -> None:
        formula = self.dense_forward_formula(layer_name, activation, layer_idx=layer_idx)
        self.render(formula, title=f"→ Forward: {layer_name}")

    def render_backward(self, layer_name: str, activation: str,
                        layer_idx: int) -> None:
        formula = self.dense_backward_formula(layer_name, activation, layer_idx=layer_idx)
        self.render(formula, title=f"← Backward: {layer_name}", style="yellow")

    def render_loss(self, loss_name: str) -> None:
        f_str = self.loss_formula(loss_name)
        g_str = self.loss_gradient_formula(loss_name)
        self.render(f"{f_str}\n{g_str}", title=f"Loss: {loss_name}", style="red")

    # ------------------------------------------------------------------
    # SymPy integration
    # ------------------------------------------------------------------

    def sympy_to_unicode(self, expr) -> str:
        """Convert a SymPy expression to pretty Unicode text."""
        if HAS_SYMPY:
            return pretty(expr, use_unicode=True)
        return str(expr)

    def sympy_to_latex(self, expr) -> str:
        """Convert a SymPy expression to LaTeX."""
        if HAS_SYMPY:
            return latex(expr)
        return str(expr)
