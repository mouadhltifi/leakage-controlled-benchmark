"""Loss functions and a factory that builds them from config.

Milestone 5 deliverable. The multi-task loss always passes
``ignore_index=-1`` for the classification branch (resolves audit
finding B.2).

Public API
----------

* :class:`MultiTaskLoss` — weighted sum across active targets.
* :class:`VolatilityLoss` — dedicated MSE on |log_return|.
* :func:`build_loss` — factory picking between the two.
* :func:`compute_class_weights` — per-class weight helper with two
  schemes (``inverse_frequency``, ``balanced``).
"""

from .factory import build_loss, compute_class_weights
from .multitask import MultiTaskLoss
from .volatility import VolatilityLoss

__all__ = [
    "MultiTaskLoss",
    "VolatilityLoss",
    "build_loss",
    "compute_class_weights",
]
