"""Structured logging setup.

Named ``logging_`` (trailing underscore) to avoid shadowing the stdlib
``logging`` module when imported from inside the package.

Library code uses ``logging.getLogger(__name__)`` directly; only the CLI
entry points call :func:`setup_logging` once to install handlers.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

_CONFIGURED = False


def setup_logging(cfg: Any | None = None, *, force: bool = False) -> None:
    """Configure the root logger once.

    Parameters
    ----------
    cfg
        An :class:`~mmfp.config.schema.ExperimentConfig` or any object
        exposing ``cfg.logging.level``. May be ``None``; then defaults
        to ``INFO``.
    force
        If ``True``, re-configure even if logging was already set up.
        Useful when tests call this repeatedly.

    Notes
    -----
    A module-level flag ensures multiple calls don't install duplicate
    handlers (which would produce double log lines). No ``print()``
    is used anywhere in library code; emit via ``logging`` instead.
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    level_name = "INFO"
    if cfg is not None and hasattr(cfg, "logging"):
        level_name = getattr(cfg.logging, "level", "INFO")
    level = getattr(logging, level_name, logging.INFO)

    # Remove any existing handlers so repeated setup doesn't double-log.
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)

    handler = logging.StreamHandler(stream=sys.stderr)
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(formatter)
    handler.setLevel(level)
    root.addHandler(handler)
    root.setLevel(level)

    _CONFIGURED = True


__all__ = ["setup_logging"]
