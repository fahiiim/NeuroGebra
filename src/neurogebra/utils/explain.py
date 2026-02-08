"""
Explanation engine for Neurogebra.

Provides detailed, educational explanations of mathematical expressions.
"""

from typing import Optional
import sympy as sp
from neurogebra.core.expression import Expression


class ExpressionExplainer:
    """
    Generates detailed explanations of mathematical expressions.

    Supports multiple difficulty levels and output formats.
    """

    @staticmethod
    def explain(
        expression: Expression,
        level: str = "intermediate",
        format: str = "text",
    ) -> str:
        """
        Generate explanation for an expression.

        Args:
            expression: Expression to explain
            level: Detail level ('beginner', 'intermediate', 'advanced')
            format: Output format ('text', 'markdown', 'latex')

        Returns:
            Explanation string
        """
        if format == "markdown":
            return ExpressionExplainer._explain_markdown(expression, level)
        elif format == "latex":
            return ExpressionExplainer._explain_latex(expression, level)
        else:
            return expression.explain(level=level)

    @staticmethod
    def _explain_markdown(expression: Expression, level: str) -> str:
        """Generate Markdown-formatted explanation."""
        lines = []
        lines.append(f"## {expression.name}")
        lines.append("")
        lines.append(f"**Formula:** ${expression.formula}$")
        lines.append("")

        if "description" in expression.metadata:
            lines.append(f"**Description:** {expression.metadata['description']}")
            lines.append("")

        if "usage" in expression.metadata:
            lines.append(f"**Usage:** {expression.metadata['usage']}")
            lines.append("")

        if expression.params:
            lines.append("**Parameters:**")
            for k, v in expression.params.items():
                lines.append(f"- `{k}` = {v}")
            lines.append("")

        if level in ("intermediate", "advanced"):
            if "pros" in expression.metadata:
                lines.append("**Advantages:**")
                for pro in expression.metadata["pros"]:
                    lines.append(f"- ✅ {pro}")
                lines.append("")

            if "cons" in expression.metadata:
                lines.append("**Disadvantages:**")
                for con in expression.metadata["cons"]:
                    lines.append(f"- ⚠️ {con}")
                lines.append("")

        if level == "advanced":
            lines.append("**Derivatives:**")
            for var in expression.variables:
                grad = sp.diff(expression.symbolic_expr, var)
                lines.append(f"- $\\frac{{\\partial}}{{\\partial {var}}} = {sp.latex(grad)}$")
            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _explain_latex(expression: Expression, level: str) -> str:
        """Generate LaTeX-formatted explanation."""
        lines = []
        lines.append(f"\\section{{{expression.name}}}")
        lines.append(f"\\[{expression.formula}\\]")
        lines.append("")

        if "description" in expression.metadata:
            lines.append(f"{expression.metadata['description']}")
            lines.append("")

        if level == "advanced":
            lines.append("\\subsection{Derivatives}")
            for var in expression.variables:
                grad = sp.diff(expression.symbolic_expr, var)
                lines.append(
                    f"\\[\\frac{{\\partial}}{{\\partial {var}}} = {sp.latex(grad)}\\]"
                )

        return "\n".join(lines)

    @staticmethod
    def compare_expressions(
        expressions: list,
        format: str = "text",
    ) -> str:
        """
        Compare multiple expressions.

        Args:
            expressions: List of Expression instances
            format: Output format

        Returns:
            Comparison string
        """
        if format == "markdown":
            lines = ["| Name | Formula | Category |", "|------|---------|----------|"]
            for expr in expressions:
                cat = expr.metadata.get("category", "N/A")
                formula = str(expr.symbolic_expr)[:30]
                lines.append(f"| {expr.name} | ${formula}$ | {cat} |")
            return "\n".join(lines)
        else:
            lines = [f"{'Name':<20} {'Formula':<40} {'Category':<15}"]
            lines.append("-" * 75)
            for expr in expressions:
                cat = expr.metadata.get("category", "N/A")
                formula = str(expr.symbolic_expr)[:38]
                lines.append(f"{expr.name:<20} {formula:<40} {cat:<15}")
            return "\n".join(lines)
