"""
MathForge: Main interface for Neurogebra.

Provides user-facing API for accessing and creating mathematical expressions.
"""

from typing import Any, Dict, List, Optional, Union
from neurogebra.core.expression import Expression
from neurogebra.repository import (
    activations,
    losses,
    regularizers,
    algebra,
    calculus,
    statistics,
    linalg,
    optimization,
    metrics,
    transforms,
)


class MathForge:
    """
    Central hub for accessing mathematical expressions.

    MathForge provides a unified interface to:
    - Get pre-built expressions
    - Create custom expressions
    - Search for expressions
    - Compose expressions
    - Train expressions

    Examples:
        >>> forge = MathForge()
        >>> relu = forge.get("relu")
        >>> result = relu.eval(x=-5)
        >>> print(result)  # 0
    """

    def __init__(self):
        """Initialize MathForge with expression repository."""
        self._repository: Dict[str, Expression] = {}
        self._load_repository()

    def _load_repository(self):
        """Load all expressions from repository modules."""
        # Load activations
        self._repository.update(activations.get_activations())

        # Load losses
        self._repository.update(losses.get_losses())

        # Load regularizers
        self._repository.update(regularizers.get_regularizers())

        # Load algebra
        self._repository.update(algebra.get_algebra_expressions())

        # Load calculus
        self._repository.update(calculus.get_calculus_expressions())

        # Load statistics
        self._repository.update(statistics.get_statistics_expressions())

        # Load linear algebra
        self._repository.update(linalg.get_linalg_expressions())

        # Load optimization
        self._repository.update(optimization.get_optimization_expressions())

        # Load metrics
        self._repository.update(metrics.get_metrics_expressions())

        # Load transforms
        self._repository.update(transforms.get_transforms_expressions())

    def get(
        self,
        name: str,
        params: Optional[Dict[str, Any]] = None,
        trainable: bool = False,
    ) -> Expression:
        """
        Get an expression by name.

        Args:
            name: Name of the expression
            params: Parameter overrides
            trainable: Whether to make parameters trainable

        Returns:
            Expression instance

        Raises:
            KeyError: If expression not found

        Examples:
            >>> forge = MathForge()
            >>> sigmoid = forge.get("sigmoid")
            >>> custom = forge.get("leaky_relu", params={"alpha": 0.2})
        """
        if name not in self._repository:
            available = ", ".join(list(self._repository.keys())[:10])
            raise KeyError(
                f"Expression '{name}' not found. "
                f"Available: {available}..."
            )

        expr_template = self._repository[name]

        # Clone and customize
        expr = Expression(
            name=expr_template.name,
            symbolic_expr=expr_template.symbolic_expr,
            params=expr_template.params.copy(),
            trainable_params=(
                list(expr_template.params.keys())
                if trainable
                else expr_template.trainable_params.copy()
            ),
            metadata=expr_template.metadata.copy(),
        )

        # Override parameters
        if params:
            expr.params.update(params)

        return expr

    def register(self, name: str, expression: Expression) -> None:
        """
        Register a custom expression in the repository.

        Args:
            name: Name for the expression
            expression: Expression instance to register
        """
        self._repository[name] = expression

    def search(self, query: str) -> List[str]:
        """
        Search for expressions by name or description.

        Args:
            query: Search string

        Returns:
            List of matching expression names
        """
        query_lower = query.lower()
        results = []

        for name, expr in self._repository.items():
            # Search in name
            if query_lower in name.lower():
                results.append(name)
                continue

            # Search in metadata
            if "description" in expr.metadata:
                if query_lower in expr.metadata["description"].lower():
                    results.append(name)
                    continue

            # Search in category
            if "category" in expr.metadata:
                if query_lower in expr.metadata["category"].lower():
                    results.append(name)

        return results

    def list_all(self, category: Optional[str] = None) -> List[str]:
        """
        List all available expressions.

        Args:
            category: Filter by category (e.g., 'activation', 'loss')

        Returns:
            List of expression names
        """
        if category is None:
            return list(self._repository.keys())

        return [
            name
            for name, expr in self._repository.items()
            if expr.metadata.get("category") == category
        ]

    def compose(self, expression_str: str, **params: Any) -> Expression:
        """
        Compose expressions using string notation.

        Args:
            expression_str: Expression like "mse + 0.1*l2"
            **params: Parameters for composed expression

        Returns:
            Composed Expression

        Examples:
            >>> forge = MathForge()
            >>> loss = forge.compose("mse + 0.1*mae")
        """
        parts = expression_str.split("+")

        result = None
        for part in parts:
            part = part.strip()

            # Handle scalar multiplication
            if "*" in part:
                scalar_str, expr_name = part.split("*", 1)
                scalar = float(scalar_str.strip())
                expr = self.get(expr_name.strip())
                expr = expr * scalar
            else:
                expr = self.get(part)

            if result is None:
                result = expr
            else:
                result = result + expr

        if result is None:
            raise ValueError(f"Could not parse expression: {expression_str}")

        return result

    def explain(self, name: str, level: str = "intermediate") -> str:
        """
        Get explanation for an expression.

        Args:
            name: Name of the expression
            level: Detail level ('beginner', 'intermediate', 'advanced')

        Returns:
            Explanation string
        """
        expr = self.get(name)
        return expr.explain(level=level)

    def compare(self, names: List[str]) -> str:
        """
        Compare multiple expressions side by side.

        Args:
            names: List of expression names to compare

        Returns:
            Comparison table as string
        """
        lines = [f"{'Name':<20} {'Formula':<40} {'Category':<15}"]
        lines.append("-" * 75)

        for name in names:
            if name in self._repository:
                expr = self._repository[name]
                formula = str(expr.symbolic_expr)[:38]
                category = expr.metadata.get("category", "N/A")
                lines.append(f"{name:<20} {formula:<40} {category:<15}")

        return "\n".join(lines)
