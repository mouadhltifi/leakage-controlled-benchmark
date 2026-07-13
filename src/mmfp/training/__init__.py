"""Training loop, callbacks, scheduler factory.

Milestone 6 deliverable. Trainer uses the seeded DataLoader generator
and worker init from :mod:`mmfp.utils.seeding`.

Public API
----------

* :class:`Trainer` — orchestrates train / validation epochs.
* :class:`History` — per-epoch training history dataclass.
* :class:`Callback` / :class:`EarlyStopping` /
  :class:`BestModelCheckpoint` / :class:`LRLogger` — pluggable
  callbacks. The trainer does not register any of them implicitly.
* :func:`build_scheduler` — factory returning the scheduler selected by
  ``cfg.training.scheduler``.
"""

from __future__ import annotations

from .callbacks import (
    BestModelCheckpoint,
    Callback,
    EarlyStopping,
    LRLogger,
    Mode,
)
from .scheduler import build_scheduler, is_plateau_scheduler
from .trainer import History, Trainer

__all__ = [
    "BestModelCheckpoint",
    "Callback",
    "EarlyStopping",
    "History",
    "LRLogger",
    "Mode",
    "Trainer",
    "build_scheduler",
    "is_plateau_scheduler",
]
