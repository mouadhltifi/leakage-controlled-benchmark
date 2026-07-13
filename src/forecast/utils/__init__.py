"""v3 utilities — thin re-exports of :mod:`mmfp.utils`.

Per the architecture spec reuse map, v3 uses v2's utilities verbatim. The
aliases here let downstream v3 modules write
``from forecast.utils import set_all_seeds`` without hard-coding the
``mmfp`` package path.

Re-exports:

* :func:`mmfp.utils.seeding.set_all_seeds`
* :func:`mmfp.utils.seeding.dataloader_worker_init`
* :func:`mmfp.utils.seeding.make_dataloader_generator`
* :func:`mmfp.utils.device.resolve_device`
* :func:`mmfp.utils.logging_.setup_logging`
* :func:`mmfp.utils.io.atomic_csv_append`
* :func:`mmfp.utils.io.save_json`
* :func:`mmfp.utils.io.get_git_sha`
"""

from __future__ import annotations

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
