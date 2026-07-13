"""Data package: raw loaders, universe constants, HDF5 store.

Per-fold assembly (``assemble_fold``) and the ``FoldArtifacts``
dataclass land in Milestone 4. This module currently exposes:

* :mod:`mmfp.data.loaders` — one loader per raw data source (M2)
* :mod:`mmfp.data.universe` — 55-ticker universe + sector map + FRED series
* :mod:`mmfp.data.paths` — canonical filesystem paths
* :mod:`mmfp.data.hdf5_store` — atomic HDF5 read/write wrapper (M2 stub)

See spec Section 3.2.
"""
