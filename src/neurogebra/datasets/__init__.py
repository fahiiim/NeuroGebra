"""
Built-in datasets for learning and experimentation.

Neurogebra now includes 100+ datasets organized by category:
- Classification (Binary & Multi-class): 25+ datasets
- Regression: 25+ datasets
- Clustering: 15+ datasets
- Time Series: 15+ datasets
- Image Recognition: 10+ datasets
- Text/NLP: 5+ datasets
- Synthetic Patterns: 20+ datasets

Use Datasets.list_all() to see all available datasets.
"""

from neurogebra.datasets.loaders import Datasets
from neurogebra.datasets.expanded_loaders import ExpandedDatasets, CombinedDatasets

# Main interface - includes all base datasets plus can access expanded ones
__all__ = ["Datasets", "ExpandedDatasets", "CombinedDatasets"]

