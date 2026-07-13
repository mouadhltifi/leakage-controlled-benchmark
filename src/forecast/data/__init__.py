"""PLACEHOLDER (M2+): v3 data package.

v3 reuses :mod:`mmfp.data` wholesale — raw loaders (prices, news, social,
macro, graph), fold assembly, universe definition — per the architecture spec reuse map.

This placeholder directory exists so the v3 import path mirrors v2's and
downstream milestones can insert v3-specific subsets (e.g. the windowed
dataset in M2) without restructuring the import surface. No Python code
lives here yet.

M2 fills: (nothing in this package — the windowed dataset lives under
:mod:`forecast.datasets`; raw loaders stay under :mod:`mmfp.data`.)
"""

from __future__ import annotations

__all__: list[str] = []
