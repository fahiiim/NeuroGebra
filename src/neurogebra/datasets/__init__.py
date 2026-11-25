"""
Built-in datasets for learning and experimentation.

48 unique educational loader methods are available through CombinedDatasets.
Many generate synthetic teaching distributions; consult the dataset catalog
for provenance and return conventions.
"""

from neurogebra.datasets.expanded_loaders import CombinedDatasets, ExpandedDatasets
from neurogebra.datasets.loaders import Datasets

# Main interface - includes all base datasets plus can access expanded ones
__all__ = ["Datasets", "ExpandedDatasets", "CombinedDatasets"]
