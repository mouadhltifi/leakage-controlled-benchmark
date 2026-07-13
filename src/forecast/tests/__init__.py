"""v3 test package root.

Test layout mirrors :mod:`mmfp.tests`:

* ``unit/``            — fast, pure-Python unit tests (M1, M2, M3).
* ``integration/``     — end-to-end experiment run (M6).
* ``reproducibility/`` — CPU bit-identical determinism (M4).
* ``leakage/``         — train-only-scaler + graph-precompute leakage (M4).
"""
