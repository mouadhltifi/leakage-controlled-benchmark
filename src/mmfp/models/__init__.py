"""Model package: encoders, aggregation, fusion, heads, losses, predictor.

Milestone 5 deliverable. Each submodule exposes ABCs and concrete
implementations for the axes of the experimental design. The top-level
:class:`Predictor` in ``predictor.py`` assembles everything from a
validated :class:`~mmfp.config.schema.ExperimentConfig`.

Public API
----------

* :class:`Predictor` — full model assembler.
* :mod:`encoders` — :class:`PriceLSTMEncoder`, :class:`PriceFFEncoder`,
  :class:`TabularFFEncoder`, :class:`GraphGATEncoder`.
* :mod:`fusion` — :class:`ConcatFusion`,
  :class:`GatedCrossAttentionFusion`,
  :class:`MultiheadAttentionFusion`, plus :func:`build_fusion`.
* :mod:`heads` — :class:`ParallelMultiTaskHead`,
  :class:`SequentialCascadeHead`, :class:`SingleTaskHead`, plus
  :func:`build_head`.
* :mod:`losses` — :class:`MultiTaskLoss`, :class:`VolatilityLoss`,
  :func:`build_loss`, :func:`compute_class_weights`.
"""

from .predictor import Predictor

__all__ = ["Predictor"]
