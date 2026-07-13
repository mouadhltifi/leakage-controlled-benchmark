"""Deterministic seeding utilities.

Addresses audit finding B.3 (reproducibility fragile in v1). Covers the
full PyTorch + NumPy + Python-random + MPS + CUDA stack plus
``PYTHONHASHSEED`` and the cuBLAS workspace environment variable that
deterministic algorithms require.
"""

from __future__ import annotations

import logging
import os
import random
from functools import partial
from typing import Callable

import numpy as np
import torch

log = logging.getLogger(__name__)


def set_all_seeds(seed: int) -> None:
    """Seed every known RNG and enable deterministic kernels.

    Call this at the start of every experiment. On CPU this gives
    bit-identical results across runs. On MPS there are known
    non-deterministic ops (notably LSTM) where we fall back to
    ``warn_only=True``. See reproducibility tests for the documented
    tolerances.

    Parameters
    ----------
    seed
        Integer seed. Applied to Python ``random``, NumPy, PyTorch (CPU,
        MPS, CUDA), and also written to ``PYTHONHASHSEED`` and
        ``CUBLAS_WORKSPACE_CONFIG`` to keep deterministic kernels happy.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch.backends.mps.is_available():
        try:
            torch.mps.manual_seed(seed)
        except (AttributeError, RuntimeError) as exc:  # pragma: no cover
            log.debug("torch.mps.manual_seed unavailable: %s", exc)

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    # cuDNN flags; no-op on MPS/CPU but cheap to set.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception as exc:  # pragma: no cover - device-dependent
        log.debug("use_deterministic_algorithms not supported: %s", exc)


def make_dataloader_generator(seed: int) -> torch.Generator:
    """Create a seeded :class:`torch.Generator` for ``DataLoader``.

    Pass as ``DataLoader(generator=make_dataloader_generator(seed), ...)``
    to fix the sampler order across runs.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def dataloader_worker_init(worker_id: int, seed: int) -> None:
    """``worker_init_fn`` for ``torch.utils.data.DataLoader``.

    Each worker gets a distinct but deterministic seed derived from the
    experiment seed so sample augmentations (if any) remain reproducible.

    Bind the ``seed`` argument with :func:`functools.partial` before
    passing to ``DataLoader``::

        from functools import partial
        worker_init_fn=partial(dataloader_worker_init, seed=cfg.seed)
    """
    worker_seed = seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def make_worker_init_fn(seed: int) -> Callable[[int], None]:
    """Convenience: return a one-arg ``worker_init_fn`` bound to ``seed``.

    ``DataLoader`` passes just the worker id to ``worker_init_fn``; this
    helper partials the seed for callers that prefer explicit binding.
    """
    return partial(dataloader_worker_init, seed=seed)


__all__ = [
    "dataloader_worker_init",
    "make_dataloader_generator",
    "make_worker_init_fn",
    "set_all_seeds",
]
