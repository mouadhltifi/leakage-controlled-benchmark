"""Device resolution with MPS > CUDA > CPU priority.

The :class:`~mmfp.config.schema.ExperimentConfig` ``device`` field may be
``"mps"``, ``"cuda"``, ``"cpu"``, or ``"auto"``. This module converts it
into a concrete :class:`torch.device`.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

log = logging.getLogger(__name__)


def _device_available(name: str) -> bool:
    if name == "mps":
        return bool(torch.backends.mps.is_available())
    if name == "cuda":
        return bool(torch.cuda.is_available())
    if name == "cpu":
        return True
    return False


def resolve_device(cfg: Any | None = None) -> torch.device:
    """Resolve a :class:`torch.device` from config.

    Parameters
    ----------
    cfg
        An :class:`~mmfp.config.schema.ExperimentConfig` or any object
        with a ``.device`` attribute in ``{"mps","cuda","cpu","auto"}``.
        May also be ``None``, in which case auto-selection is used.

    Returns
    -------
    torch.device
        * If ``cfg.device == "auto"`` (or ``cfg is None``): first
          available of ``mps`` > ``cuda`` > ``cpu``.
        * Otherwise the requested device. Missing hardware falls back to
          CPU with a warning.
    """
    requested = "auto"
    if cfg is not None and hasattr(cfg, "device"):
        requested = cfg.device

    if requested == "auto":
        for candidate in ("mps", "cuda", "cpu"):
            if _device_available(candidate):
                log.info("resolve_device: auto-selected %s", candidate)
                return torch.device(candidate)
        # Unreachable but defensive.
        return torch.device("cpu")  # pragma: no cover

    if not _device_available(requested):
        log.warning(
            "resolve_device: requested %r unavailable, falling back to cpu",
            requested,
        )
        return torch.device("cpu")
    return torch.device(requested)


__all__ = ["resolve_device"]
