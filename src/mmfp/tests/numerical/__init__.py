"""Numerical-stability tests (spec Section 5.5).

Guarantees critical corner cases behave correctly — zero vectors,
negative inputs to monotone transforms, extreme values, etc. — where
a silent ``NaN`` or ``Inf`` downstream would corrupt a run.
"""
