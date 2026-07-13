"""Prediction heads: parallel multitask, sequential cascade, single-task.

Milestone 5 deliverable. Implements axis 1 (architecture) of the
experimental design. Output is a dict keyed by target name.

Public classes
--------------

* :class:`ParallelMultiTaskHead` — independent per-target MLPs.
* :class:`SequentialCascadeHead` — forecast-then-classify with joint
  (or detached) gradients.
* :class:`SingleTaskHead` — one MLP for a single target.

Use :func:`build_head` to instantiate the one named in
``cfg.head.architecture``.
"""

from __future__ import annotations

from mmfp.config.schema import ExperimentConfig

from .base import PredictionHead
from .parallel_multitask import ParallelMultiTaskHead
from .sequential_cascade import SequentialCascadeHead
from .single_task import SingleTaskHead


def build_head(cfg: ExperimentConfig) -> PredictionHead:
    """Instantiate the head selected by ``cfg.head.architecture``.

    Parameters
    ----------
    cfg
        Validated experiment config. The cross-field validator must
        have run first so target / architecture / cascade combinations
        are already verified.

    Returns
    -------
    PredictionHead
        A concrete head ready to be called on a ``(B, H)`` tensor.

    Raises
    ------
    ValueError
        If ``cfg.head.architecture`` is not one of the three supported
        values.
    """
    architecture = cfg.head.architecture
    if architecture == "parallel_multitask":
        return ParallelMultiTaskHead(cfg)
    if architecture == "sequential_cascade":
        return SequentialCascadeHead(cfg)
    if architecture == "single_task":
        return SingleTaskHead(cfg)
    raise ValueError(f"Unknown head architecture: {architecture!r}")


__all__ = [
    "ParallelMultiTaskHead",
    "PredictionHead",
    "SequentialCascadeHead",
    "SingleTaskHead",
    "build_head",
]
