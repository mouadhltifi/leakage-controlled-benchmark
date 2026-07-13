"""Cross-cutting utilities: seeding, device resolution, logging, IO.

Contains no model, data, or feature logic. Everything here must be
importable without pulling in heavy dependencies beyond ``torch`` and
``numpy``.
"""

from mmfp.utils.device import resolve_device
from mmfp.utils.io import atomic_csv_append, get_git_sha, save_json
from mmfp.utils.logging_ import setup_logging
from mmfp.utils.seeding import (
    dataloader_worker_init,
    make_dataloader_generator,
    set_all_seeds,
)

__all__ = [
    "atomic_csv_append",
    "dataloader_worker_init",
    "get_git_sha",
    "make_dataloader_generator",
    "resolve_device",
    "save_json",
    "set_all_seeds",
    "setup_logging",
]
