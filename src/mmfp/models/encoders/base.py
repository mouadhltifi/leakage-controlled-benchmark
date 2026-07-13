"""Abstract base class shared by every encoder.

Every encoder in :mod:`mmfp.models.encoders` consumes exactly one
modality's raw tensor and emits a dense ``(B, H)`` vector (or ``(N, H)``
for node-level graph outputs). The contract is a compile-time invariant
that :class:`~mmfp.models.predictor.Predictor` relies on to wire modality
encodings into the fusion layer without extra reshaping.

Contract
--------

* ``output_dim`` attribute: always equals ``cfg.model.hidden_dim`` so
  callers can size downstream layers without introspecting the module.
* ``cfg.model.dropout`` is the single dropout rate used internally;
  encoders never hardcode a different value.
* ``nn.LayerNorm(H)`` is the final op before return. Keeps fusion input
  in a well-scaled regime regardless of the upstream distribution.
* Forward passes are deterministic given seeded RNG state. Avoid
  ``torch.bernoulli`` outside ``nn.Dropout`` (which respects
  ``torch.use_deterministic_algorithms``) and any PyTorch op that
  PyTorch's determinism guide flags as nondeterministic.

The base class is intentionally thin; subclasses must implement
``__init__`` and ``forward``. ``EncoderBase`` only records the output
dim and defines the ABC.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn


class EncoderBase(nn.Module, ABC):
    """Abstract base for all modality encoders.

    Subclasses MUST:

    * accept an :class:`~mmfp.config.schema.ExperimentConfig` or the
      relevant sub-config plus the per-modality input dim;
    * set ``self.output_dim = cfg.model.hidden_dim`` so downstream
      layers can size linear projections without looking inside;
    * apply ``nn.LayerNorm(self.output_dim)`` as the last op;
    * read every dropout probability from ``cfg.model.dropout`` — no
      hardcoded values.

    Parameters
    ----------
    output_dim
        The dense representation dimension. Always equal to
        ``cfg.model.hidden_dim``; exposed as an attribute so callers can
        introspect without reaching into the config.
    """

    def __init__(self, output_dim: int) -> None:
        super().__init__()
        if output_dim < 1:
            raise ValueError(
                f"output_dim must be a positive integer; got {output_dim}"
            )
        self.output_dim: int = output_dim

    @abstractmethod
    def forward(self, *args, **kwargs) -> Tensor:  # pragma: no cover - abstract
        """Subclass-specific forward pass.

        Concrete encoders accept their modality-specific tensor shapes.
        The output is always ``(B, H)`` for batch-level encoders or
        ``(N, H)`` for node-level graph encoders, where ``H ==
        self.output_dim``.
        """
        ...


__all__ = ["EncoderBase"]
