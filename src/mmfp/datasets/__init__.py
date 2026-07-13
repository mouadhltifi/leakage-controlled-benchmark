"""PyTorch ``Dataset`` wrappers over pre-assembled fold artifacts.

Milestone 4 deliverable. Thin indexing layer over the arrays returned
by :mod:`mmfp.data.assemble`; performs no feature engineering itself.
"""

from mmfp.datasets.collate import make_collate_fn
from mmfp.datasets.multimodal import MultiModalDataset, Split

__all__ = ["MultiModalDataset", "Split", "make_collate_fn"]
