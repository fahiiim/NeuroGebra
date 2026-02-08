"""Tests for search functionality."""

import pytest
from neurogebra.repository.search import ExpressionSearch
from neurogebra.repository.activations import get_activations
from neurogebra.repository.losses import get_losses


@pytest.fixture
def search_engine():
    """Create search engine with all repository expressions."""
    repo = {}
    repo.update(get_activations())
    repo.update(get_losses())
    return ExpressionSearch(repo)


class TestExpressionSearch:
    """Test search functionality."""

    def test_exact_name_search(self, search_engine):
        """Test exact name search returns best match."""
        results = search_engine.search("relu")
        assert len(results) > 0
        assert results[0]["name"] == "relu"

    def test_partial_name_search(self, search_engine):
        """Test partial name search."""
        results = search_engine.search("sig")
        names = [r["name"] for r in results]
        assert "sigmoid" in names

    def test_description_search(self, search_engine):
        """Test searching by description."""
        results = search_engine.search("regression")
        assert len(results) > 0

    def test_category_filter(self, search_engine):
        """Test search with category filter."""
        results = search_engine.search("", category="activation")
        for r in results:
            assert r["category"] == "activation"

    def test_max_results(self, search_engine):
        """Test max results limit."""
        results = search_engine.search("", max_results=3)
        assert len(results) <= 3

    def test_get_categories(self, search_engine):
        """Test getting all categories."""
        categories = search_engine.get_categories()
        assert "activation" in categories
        assert "loss" in categories

    def test_get_by_category(self, search_engine):
        """Test getting expressions by category."""
        acts = search_engine.get_by_category("activation")
        assert "relu" in acts
        assert "sigmoid" in acts

    def test_no_results(self, search_engine):
        """Test search with no matching results."""
        results = search_engine.search("zzzznoexist12345")
        assert len(results) == 0
