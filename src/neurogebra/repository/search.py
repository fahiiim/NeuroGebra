"""
Search utilities for expression repository.

Provides advanced search and filtering capabilities.
"""

from typing import Dict, List, Optional
from neurogebra.core.expression import Expression


class ExpressionSearch:
    """
    Advanced search engine for mathematical expressions.

    Supports fuzzy matching, category filtering, and similarity search.
    """

    def __init__(self, repository: Dict[str, Expression]):
        """
        Initialize search engine.

        Args:
            repository: Dictionary of expression name -> Expression
        """
        self._repository = repository

    def search(
        self,
        query: str,
        category: Optional[str] = None,
        max_results: int = 10,
    ) -> List[Dict]:
        """
        Search for expressions matching query.

        Args:
            query: Search string
            category: Optional category filter
            max_results: Maximum number of results

        Returns:
            List of matching expression info dicts
        """
        query_lower = query.lower()
        results = []

        for name, expr in self._repository.items():
            # Category filter
            if category and expr.metadata.get("category") != category:
                continue

            score = self._compute_score(query_lower, name, expr)
            if score > 0:
                results.append(
                    {
                        "name": name,
                        "score": score,
                        "category": expr.metadata.get("category", ""),
                        "description": expr.metadata.get("description", ""),
                        "formula": str(expr.symbolic_expr),
                    }
                )

        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:max_results]

    def _compute_score(
        self, query: str, name: str, expr: Expression
    ) -> float:
        """
        Compute relevance score for a query-expression pair.

        Args:
            query: Lowercase search query
            name: Expression name
            expr: Expression instance

        Returns:
            Relevance score (0 = no match)
        """
        score = 0.0

        # Exact name match
        if query == name.lower():
            score += 10.0

        # Partial name match
        elif query in name.lower():
            score += 5.0

        # Name starts with query
        if name.lower().startswith(query):
            score += 3.0

        # Description match
        description = expr.metadata.get("description", "").lower()
        if query in description:
            score += 2.0

        # Usage match
        usage = expr.metadata.get("usage", "").lower()
        if query in usage:
            score += 1.0

        # Keyword match
        query_words = query.split()
        for word in query_words:
            if word in name.lower():
                score += 0.5
            if word in description:
                score += 0.3

        return score

    def get_categories(self) -> List[str]:
        """
        Get all available categories.

        Returns:
            List of unique category names
        """
        categories = set()
        for expr in self._repository.values():
            cat = expr.metadata.get("category")
            if cat:
                categories.add(cat)
        return sorted(categories)

    def get_by_category(self, category: str) -> List[str]:
        """
        Get all expression names in a category.

        Args:
            category: Category name

        Returns:
            List of expression names
        """
        return [
            name
            for name, expr in self._repository.items()
            if expr.metadata.get("category") == category
        ]
