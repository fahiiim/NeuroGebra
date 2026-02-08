"""
NeuroCraft: Enhanced mathematical expression interface.

The main hub for accessing, creating, and learning about mathematical
operations used in machine learning and deep learning.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
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


class NeuroCraft:
    """
    Central interface for mathematical expressions and learning.

    NeuroCraft provides:
    - Access to pre-built mathematical expressions
    - Interactive learning tools
    - Model building capabilities
    - Expression composition
    - Training utilities

    Examples:
        >>> craft = NeuroCraft(educational_mode=False)
        >>> relu = craft.get("relu")
        >>> result = relu.eval(x=5)
    """

    def __init__(self, educational_mode: bool = True):
        """
        Initialize NeuroCraft.

        Args:
            educational_mode: Enable educational features (explanations, tips)
        """
        self.educational_mode = educational_mode
        self._repository: Dict[str, Expression] = {}
        self._load_repository()

        if educational_mode:
            self._show_welcome()

    def _load_repository(self):
        """Load all expressions from repository modules."""
        self._repository.update(activations.get_activations())
        self._repository.update(losses.get_losses())
        self._repository.update(regularizers.get_regularizers())
        self._repository.update(algebra.get_algebra_expressions())
        self._repository.update(calculus.get_calculus_expressions())
        self._repository.update(statistics.get_statistics_expressions())
        self._repository.update(linalg.get_linalg_expressions())
        self._repository.update(optimization.get_optimization_expressions())
        self._repository.update(metrics.get_metrics_expressions())
        self._repository.update(transforms.get_transforms_expressions())

    def _show_welcome(self):
        """Show welcome message for beginners."""
        print("🎓 Welcome to Neurogebra!")
        print("   Type craft.tutorial() to start learning")
        print("   Type craft.search('activation') to explore")

    def _find_similar(self, name: str, max_results: int = 5) -> List[str]:
        """
        Find expression names similar to the given name.

        Args:
            name: Name to search for
            max_results: Maximum suggestions to return

        Returns:
            List of similar expression names
        """
        name_lower = name.lower()
        scored = []
        for key in self._repository:
            key_lower = key.lower()
            # Simple similarity: shared substring length
            score = 0
            for i in range(len(name_lower)):
                for j in range(i + 1, len(name_lower) + 1):
                    sub = name_lower[i:j]
                    if sub in key_lower:
                        score = max(score, len(sub))
            if score > 0:
                scored.append((key, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [k for k, _ in scored[:max_results]]

    def get(
        self,
        name: str,
        params: Optional[Dict[str, Any]] = None,
        explain: bool = False,
    ) -> Expression:
        """
        Get a mathematical expression by name.

        Args:
            name: Name of expression (e.g. 'relu', 'sigmoid', 'mse')
            params: Custom parameter overrides
            explain: Show explanation immediately

        Returns:
            Expression instance

        Examples:
            >>> craft = NeuroCraft(educational_mode=False)
            >>> relu = craft.get("relu")
            >>> relu.eval(x=5)
            5
        """
        if name not in self._repository:
            suggestions = self._find_similar(name)
            hint = ""
            if suggestions:
                hint = f"\n   Did you mean: {', '.join(suggestions)}?"
            raise KeyError(
                f"Expression '{name}' not found.{hint}\n"
                f"   Use craft.search('{name}') to find related expressions."
            )

        expr = self._repository[name].clone()

        if params:
            expr.params.update(params)

        if explain and self.educational_mode:
            print(expr.explain())

        return expr

    def search(
        self,
        query: str,
        category: Optional[str] = None,
        show_details: bool = True,
    ) -> List[str]:
        """
        Search for expressions with smart suggestions.

        Args:
            query: Search term (e.g. 'relu', 'loss', 'activation')
            category: Filter by category ('activation', 'loss', etc.)
            show_details: Show descriptions in output

        Returns:
            List of matching expression names
        """
        results = []

        for name, expr in self._repository.items():
            # Category filter
            if category and expr.metadata.get("category") != category:
                continue

            # Search in name
            if query.lower() in name.lower():
                results.append((name, expr))
            elif "description" in expr.metadata:
                if query.lower() in expr.metadata["description"].lower():
                    results.append((name, expr))

        if show_details and self.educational_mode and results:
            self._display_search_results(results)

        return [name for name, _ in results]

    def _display_search_results(self, results: List[tuple]):
        """Display search results with formatting."""
        print(f"\n📚 Found {len(results)} expressions:\n")

        for name, expr in results[:10]:
            category = expr.metadata.get("category", "general")
            desc = expr.metadata.get("description", "No description")
            print(f"  • {name} [{category}]")
            print(f"    {desc[:80]}")
            print()

    def list_all(self, category: Optional[str] = None) -> List[str]:
        """
        List all available expressions.

        Args:
            category: Optional filter by category

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

    def register(self, name: str, expression: Expression) -> None:
        """
        Register a custom expression in the repository.

        Args:
            name: Name for the expression
            expression: Expression instance to register
        """
        self._repository[name] = expression

    def compose(self, expression_str: str, **params) -> Expression:
        """
        Compose expressions using string notation.

        Args:
            expression_str: Expression like "mse + 0.1*mae"
            **params: Parameters for composed expression

        Returns:
            Composed Expression

        Examples:
            >>> craft = NeuroCraft(educational_mode=False)
            >>> loss = craft.compose("mse + 0.1*mae")
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

    def compare(
        self,
        expressions: List[str],
        on_data: Optional[tuple] = None,
        metric: str = "behavior",
    ):
        """
        Compare multiple expressions visually.

        Args:
            expressions: List of expression names to compare
            on_data: Optional (X, y) data for evaluation
            metric: What to compare ('behavior', 'gradient', 'performance')
        """
        from neurogebra.viz.plotting import plot_comparison

        exprs = [self.get(name) for name in expressions]

        if metric == "behavior":
            plot_comparison(exprs, title="Expression Behavior Comparison")
        elif metric == "gradient":
            grad_exprs = [e.gradient("x") for e in exprs]
            plot_comparison(
                grad_exprs, title="Gradient Comparison"
            )

    def tutorial(self, topic: Optional[str] = None):
        """
        Start an interactive tutorial.

        Args:
            topic: Specific topic or None for menu.
                   Options: 'basics', 'first_model', 'activations', 'training'
        """
        from neurogebra.tutorials.tutorial_system import TutorialSystem

        tutorial_system = TutorialSystem()

        if topic is None:
            tutorial_system.show_menu()
        else:
            tutorial_system.start(topic)

    def quick_activation(self, name: str = "relu") -> Expression:
        """
        Get an activation quickly with optional visualization.

        Args:
            name: Activation name (default 'relu')

        Returns:
            Expression instance
        """
        act = self.get(name)
        if self.educational_mode:
            print(act.explain())
        return act

    def quick_loss(self, name: str = "mse") -> Expression:
        """
        Get a loss function quickly with optional explanation.

        Args:
            name: Loss function name (default 'mse')

        Returns:
            Expression instance
        """
        loss = self.get(name)
        if self.educational_mode:
            print(loss.explain())
        return loss
