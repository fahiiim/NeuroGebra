"""
Core Expression class for Neurogebra.

This module defines the fundamental Expression class that represents
mathematical operations with symbolic, numerical, and trainable capabilities.
"""

from typing import Any, Optional, Union

import numpy as np
import sympy as sp
from sympy import Symbol, lambdify, sympify


class Expression:
    """
    Unified mathematical expression supporting symbolic, numerical,
    and trainable operations.

    Attributes:
        name: Human-readable name of the expression
        symbolic_expr: SymPy symbolic representation
        params: Dictionary of parameters
        trainable_params: List of parameter names that can be trained
        metadata: Additional information about the expression
    """

    def __init__(
        self,
        name: str,
        symbolic_expr: Union[str, sp.Expr],
        params: Optional[dict[str, Any]] = None,
        trainable_params: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ):
        """
        Initialize an Expression.

        Args:
            name: Name identifier for the expression
            symbolic_expr: Symbolic mathematical expression (string or SymPy)
            params: Dictionary of parameter values
            trainable_params: List of parameters that can be trained
            metadata: Additional information (description, usage, etc.)
        """
        self.name = name
        self.params = params or {}
        self.trainable_params = trainable_params or []
        self.metadata = metadata or {}

        # Convert string to SymPy expression
        if isinstance(symbolic_expr, str):
            self.symbolic_expr = sympify(symbolic_expr)
        else:
            self.symbolic_expr = symbolic_expr

        # Extract variables from symbolic expression
        self.variables = sorted(self.symbolic_expr.free_symbols, key=lambda s: s.name)
        self._var_names = [s.name for s in self.variables]

        # Create numerical function
        self._numerical_func = None
        self._c_func = None
        self._cuda_func = None
        self._preferred_backend = "numpy"
        self._compile_numerical()

    def _compile_numerical(self):
        """Compile symbolic expression to numerical function."""
        if self.variables:
            self._numerical_func = lambdify(
                self.variables, self.symbolic_expr, modules=["numpy"]
            )
        else:
            # Constant expression
            try:
                const_val = float(self.symbolic_expr)
            except Exception:
                const_val = 0.0
            self._numerical_func = lambda *args, **kwargs: const_val

    def compile(self, backend: str = "c") -> "Expression":
        """
        Pre-compile the expression into high-performance native machine code or CUDA.

        Args:
            backend: 'c' (AVX2 SIMD), 'cuda' (NVIDIA GPU), or 'numpy'

        Returns:
            Self for chaining
        """
        b = backend.lower()
        if b == "c":
            from neurogebra.acceleration.c_jit import compile_c, is_c_available

            if not is_c_available():
                raise RuntimeError("No C/C++ compiler found on system.")
            self._c_func = compile_c(self.symbolic_expr, self.variables)
            self._preferred_backend = "c"
        elif b == "cuda":
            from neurogebra.acceleration.cuda_jit import compile_cuda, is_cuda_available

            if not is_cuda_available():
                raise RuntimeError("CUDA is not available on this machine.")
            self._cuda_func = compile_cuda(self.symbolic_expr, self.variables)
            self._preferred_backend = "cuda"
        elif b == "numpy":
            self._compile_numerical()
            self._preferred_backend = "numpy"
        else:
            raise ValueError(
                f"Unknown backend '{backend}'. Supported: 'c', 'cuda', 'numpy'"
            )
        return self

    def eval(
        self, *args: Any, backend: Optional[str] = None, **kwargs: Any
    ) -> Union[float, np.ndarray]:
        """
        Evaluate the expression numerically or with hardware acceleration.

        Args:
            *args: Positional arguments for variables
            backend: Optional backend override: 'numpy', 'c', 'cuda', or 'auto'
            **kwargs: Keyword arguments for variables

        Returns:
            Numerical result (float or numpy array)

        Examples:
            >>> expr = Expression("quadratic", "a*x**2 + b*x + c")
            >>> result = expr.eval(x=2, a=1, b=2, c=1)
        """
        if not self.variables:
            return float(self.symbolic_expr)

        # Map variable values
        ordered_args = []
        pos_idx = 0

        # Identify variables not present in params
        runtime_vars = [v for v in self.variables if v.name not in self.params]

        for var in self.variables:
            name = var.name
            if name in kwargs:
                ordered_args.append(kwargs[name])
            elif name in self.params:
                ordered_args.append(self.params[name])
            elif args:
                if len(args) == len(self.variables):
                    ordered_args.append(args[self.variables.index(var)])
                elif pos_idx < len(args) and var in runtime_vars:
                    ordered_args.append(args[pos_idx])
                    pos_idx += 1
                elif pos_idx < len(args):
                    ordered_args.append(args[pos_idx])
                    pos_idx += 1
                else:
                    ordered_args.append(0.0)
            else:
                ordered_args.append(0.0)

        chosen_backend = backend.lower() if backend else self._preferred_backend

        # Dispatch to backend
        if chosen_backend == "c":
            if self._c_func is None:
                from neurogebra.acceleration.c_jit import compile_c, is_c_available

                if is_c_available():
                    self._c_func = compile_c(self.symbolic_expr, self.variables)
            if self._c_func is not None:
                return self._c_func(*ordered_args)

        elif chosen_backend == "cuda":
            if self._cuda_func is None:
                from neurogebra.acceleration.cuda_jit import (
                    compile_cuda,
                    is_cuda_available,
                )

                if is_cuda_available():
                    self._cuda_func = compile_cuda(self.symbolic_expr, self.variables)
            if self._cuda_func is not None:
                return self._cuda_func(*ordered_args)

        # Default / Fallback to fast compiled NumPy lambdify
        if self._numerical_func is None:
            self._compile_numerical()

        result = self._numerical_func(*ordered_args)
        if isinstance(result, np.ndarray) and result.shape == ():
            return float(result)
        return result

    def gradient(self, var: Union[str, Symbol]) -> "Expression":
        """
        Compute symbolic gradient with respect to a variable.

        Args:
            var: Variable to differentiate with respect to

        Returns:
            New Expression representing the gradient
        """
        if isinstance(var, str):
            var = Symbol(var)

        grad_expr = sp.diff(self.symbolic_expr, var)

        return Expression(
            name=f"d({self.name})/d({var})",
            symbolic_expr=grad_expr,
            params=self.params.copy(),
            metadata={"parent": self.name, "gradient_var": str(var)},
        )

    def compose(self, other: "Expression") -> "Expression":
        """
        Compose two expressions: self(other(x)).

        Args:
            other: Expression to compose with

        Returns:
            New composed Expression
        """
        # Simple composition: self(other)
        if not self.variables:
            return Expression(
                name=f"{self.name}∘{other.name}",
                symbolic_expr=self.symbolic_expr,
                params={**self.params, **other.params},
                metadata={"composition": [self.name, other.name]},
            )

        composed_expr = self.symbolic_expr.subs(
            {self.variables[0]: other.symbolic_expr}
        )

        return Expression(
            name=f"{self.name}∘{other.name}",
            symbolic_expr=composed_expr,
            params={**self.params, **other.params},
            metadata={"composition": [self.name, other.name]},
        )

    def clone(self) -> "Expression":
        """
        Create a deep copy of this expression.

        Returns:
            New Expression with copied attributes
        """
        return Expression(
            name=self.name,
            symbolic_expr=self.symbolic_expr,
            params=self.params.copy(),
            trainable_params=self.trainable_params.copy(),
            metadata=self.metadata.copy(),
        )

    def visualize(
        self,
        x_range: tuple[float, float] = (-5, 5),
        n_points: int = 500,
        interactive: bool = False,
        **kwargs: Any,
    ):
        """
        Visualize this expression.

        Args:
            x_range: Range for x-axis
            n_points: Number of points
            interactive: Use interactive plotly plot
            **kwargs: Additional plot parameters
        """
        from neurogebra.viz.plotting import plot_expression

        return plot_expression(self, x_range=x_range, n_points=n_points, **kwargs)

    def __call__(self, *args, **kwargs):
        """Allow expression to be called like a function."""
        return self.eval(*args, **kwargs)

    def __add__(self, other):
        """Add two expressions."""
        if isinstance(other, Expression):
            new_expr = self.symbolic_expr + other.symbolic_expr
            new_name = f"({self.name}+{other.name})"
            new_params = {**self.params, **other.params}
        else:
            new_expr = self.symbolic_expr + other
            new_name = f"({self.name}+{other})"
            new_params = self.params.copy()

        return Expression(new_name, new_expr, new_params)

    def __radd__(self, other):
        """Right addition."""
        if isinstance(other, (int, float)):
            new_expr = other + self.symbolic_expr
            new_name = f"({other}+{self.name})"
            return Expression(new_name, new_expr, self.params.copy())
        return NotImplemented

    def __sub__(self, other):
        """Subtract two expressions."""
        if isinstance(other, Expression):
            new_expr = self.symbolic_expr - other.symbolic_expr
            new_name = f"({self.name}-{other.name})"
            new_params = {**self.params, **other.params}
        else:
            new_expr = self.symbolic_expr - other
            new_name = f"({self.name}-{other})"
            new_params = self.params.copy()

        return Expression(new_name, new_expr, new_params)

    def __mul__(self, other):
        """Multiply two expressions."""
        if isinstance(other, Expression):
            new_expr = self.symbolic_expr * other.symbolic_expr
            new_name = f"({self.name}*{other.name})"
            new_params = {**self.params, **other.params}
        else:
            new_expr = self.symbolic_expr * other
            new_name = f"({self.name}*{other})"
            new_params = self.params.copy()

        return Expression(new_name, new_expr, new_params)

    def __rmul__(self, other):
        """Right multiplication."""
        if isinstance(other, (int, float)):
            new_expr = other * self.symbolic_expr
            new_name = f"({other}*{self.name})"
            return Expression(new_name, new_expr, self.params.copy())
        return NotImplemented

    def __truediv__(self, other):
        """Divide two expressions."""
        if isinstance(other, Expression):
            new_expr = self.symbolic_expr / other.symbolic_expr
            new_name = f"({self.name}/{other.name})"
            new_params = {**self.params, **other.params}
        else:
            new_expr = self.symbolic_expr / other
            new_name = f"({self.name}/{other})"
            new_params = self.params.copy()

        return Expression(new_name, new_expr, new_params)

    def __pow__(self, other):
        """Power of expression."""
        if isinstance(other, Expression):
            new_expr = self.symbolic_expr**other.symbolic_expr
            new_name = f"({self.name}**{other.name})"
            new_params = {**self.params, **other.params}
        else:
            new_expr = self.symbolic_expr**other
            new_name = f"({self.name}**{other})"
            new_params = self.params.copy()

        return Expression(new_name, new_expr, new_params)

    def __neg__(self):
        """Negate expression."""
        return Expression(f"(-{self.name})", -self.symbolic_expr, self.params.copy())

    def __repr__(self):
        return f"Expression('{self.name}': {self.symbolic_expr})"

    def __str__(self):
        return str(self.symbolic_expr)

    @property
    def formula(self) -> str:
        """Get LaTeX representation of the expression."""
        return sp.latex(self.symbolic_expr)

    def simplify(self) -> "Expression":
        """
        Return a simplified version of the expression.

        Returns:
            New simplified Expression
        """
        simplified = sp.simplify(self.symbolic_expr)
        return Expression(
            name=f"simplified({self.name})",
            symbolic_expr=simplified,
            params=self.params.copy(),
            trainable_params=self.trainable_params.copy(),
            metadata=self.metadata.copy(),
        )

    def expand(self) -> "Expression":
        """
        Return an expanded version of the expression.

        Returns:
            New expanded Expression
        """
        expanded = sp.expand(self.symbolic_expr)
        return Expression(
            name=f"expanded({self.name})",
            symbolic_expr=expanded,
            params=self.params.copy(),
            trainable_params=self.trainable_params.copy(),
            metadata=self.metadata.copy(),
        )

    def integrate(self, var: Union[str, Symbol]) -> "Expression":
        """
        Compute symbolic integral with respect to a variable.

        Args:
            var: Variable to integrate with respect to

        Returns:
            New Expression representing the integral
        """
        if isinstance(var, str):
            var = Symbol(var)

        integral_expr = sp.integrate(self.symbolic_expr, var)

        return Expression(
            name=f"∫({self.name})d{var}",
            symbolic_expr=integral_expr,
            params=self.params.copy(),
            metadata={"parent": self.name, "integral_var": str(var)},
        )

    def explain(self, level: str = "intermediate") -> str:
        """
        Provide explanation of the expression.

        Args:
            level: Explanation level ('beginner', 'intermediate', 'advanced')

        Returns:
            Explanatory text
        """
        explanation = f"Expression: {self.name}\n"
        explanation += f"Formula: {self.formula}\n"
        explanation += f"Variables: {[str(v) for v in self.variables]}\n"

        if self.params:
            explanation += f"Parameters: {self.params}\n"

        if self.trainable_params:
            explanation += f"Trainable: {self.trainable_params}\n"

        if "description" in self.metadata:
            explanation += f"\nDescription: {self.metadata['description']}\n"

        if "usage" in self.metadata:
            explanation += f"\nUsage: {self.metadata['usage']}\n"

        if level in ("intermediate", "advanced"):
            if "pros" in self.metadata:
                explanation += f"\nPros: {', '.join(self.metadata['pros'])}\n"
            if "cons" in self.metadata:
                explanation += f"Cons: {', '.join(self.metadata['cons'])}\n"

        if level == "advanced":
            # Show gradient information
            for var in self.variables:
                grad = sp.diff(self.symbolic_expr, var)
                explanation += f"\n∂/∂{var} = {grad}\n"

        return explanation
