"""
Core Expression class for Neurogebra.

This module defines the fundamental Expression class that represents
mathematical operations with symbolic, numerical, and trainable capabilities.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import sympy as sp
from sympy import Symbol, sympify, lambdify


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
        params: Optional[Dict[str, Any]] = None,
        trainable_params: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
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
        self.variables = sorted(
            list(self.symbolic_expr.free_symbols), key=lambda s: s.name
        )

        # Create numerical function
        self._numerical_func = None
        self._compile_numerical()

    def _compile_numerical(self):
        """Compile symbolic expression to numerical function."""
        if self.variables:
            self._numerical_func = lambdify(
                self.variables, self.symbolic_expr, modules=["numpy"]
            )
        else:
            # Constant expression
            self._numerical_func = lambda: float(self.symbolic_expr)

    def eval(self, *args: Any, **kwargs: Any) -> Union[float, np.ndarray]:
        """
        Evaluate the expression numerically.

        Args:
            *args: Positional arguments for variables
            **kwargs: Keyword arguments for variables

        Returns:
            Numerical result (float or numpy array)

        Examples:
            >>> expr = Expression("quadratic", "a*x**2 + b*x + c")
            >>> result = expr.eval(x=2, a=1, b=2, c=1)
        """
        # Substitute parameters into expression
        expr_with_params = self.symbolic_expr.subs(self.params)

        # Determine remaining free symbols after parameter substitution
        remaining_vars = sorted(
            list(expr_with_params.free_symbols), key=lambda s: s.name
        )

        if not remaining_vars:
            return float(expr_with_params)

        # Create function with current parameters
        func = lambdify(remaining_vars, expr_with_params, modules=["numpy"])

        # Handle both positional and keyword arguments
        if args:
            return func(*args)
        elif kwargs:
            ordered_args = [kwargs.get(str(var), 0) for var in remaining_vars]
            return func(*ordered_args)
        else:
            return func()

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
        x_range: Tuple[float, float] = (-5, 5),
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
            new_expr = self.symbolic_expr ** other.symbolic_expr
            new_name = f"({self.name}**{other.name})"
            new_params = {**self.params, **other.params}
        else:
            new_expr = self.symbolic_expr ** other
            new_name = f"({self.name}**{other})"
            new_params = self.params.copy()

        return Expression(new_name, new_expr, new_params)

    def __neg__(self):
        """Negate expression."""
        return Expression(
            f"(-{self.name})", -self.symbolic_expr, self.params.copy()
        )

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
