"""Config-aware batch collation.

Default ``torch.utils.data.default_collate`` already stacks most keys
correctly; two cases require surgery:

1. **Graph batches.** PyTorch Geometric's convention batches multiple
   graphs into one big graph by concatenating node features and offsetting
   edge indices by ``batch_position * N_nodes``. Our per-sample
   ``graph_features`` is ``(N, F)`` and ``edge_index`` is ``(2, E)``; the
   batched versions become ``(B*N, F)`` and ``(2, B*E)`` with the offset
   applied.
2. **Static + dynamic graphs.** When both edge indices are present, each
   is offset independently before concatenation.

All other modality tensors (``price``, ``macro``, ``news``, ``social``,
target tensors, ``stock_idx``) stack via :func:`torch.stack`.
"""

from __future__ import annotations

from typing import Callable, Iterable

import torch
from torch.utils.data.dataloader import default_collate

from mmfp.config.schema import ExperimentConfig
from mmfp.data.universe import N_STOCKS

#: Keys whose values are scalars / 1-D tensors, stackable by default.
_STACKABLE_KEYS: tuple[str, ...] = (
    "price",
    "macro",
    "news",
    "social",
    "cls_target",
    "reg_target",
    "vol_target",
    "stock_idx",
)

#: Keys representing per-sample node feature tensors — need BG-style concat.
_GRAPH_NODE_KEYS: tuple[str, ...] = ("graph_features",)

#: Edge-index keys that require per-sample offsetting before concat.
_EDGE_INDEX_KEYS: tuple[str, ...] = (
    "edge_index",
    "edge_index_static",
    "edge_index_dynamic",
)


def make_collate_fn(cfg: ExperimentConfig) -> Callable[[list[dict]], dict]:
    """Build a collate function honoring the config's graph settings.

    The returned callable is intended for ``DataLoader(collate_fn=...)``.

    Parameters
    ----------
    cfg
        Experiment configuration. Only ``cfg.graph.enabled`` and
        ``cfg.graph.source`` affect the collate behaviour.

    Returns
    -------
    Callable[[list[dict]], dict]
        A function that accepts a list of ``__getitem__`` outputs and
        returns a batched dict.
    """
    graph_enabled = cfg.graph.enabled
    n_nodes = N_STOCKS

    def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        if not batch:
            raise ValueError("Empty batch passed to collate_fn")

        keys = batch[0].keys()
        out: dict[str, torch.Tensor] = {}

        # 1. Stack the flat-tensor keys.
        for key in _STACKABLE_KEYS:
            if key in keys:
                out[key] = torch.stack([b[key] for b in batch], dim=0)

        # 2. Graph-specific handling.
        if graph_enabled:
            # Node features → concat along node axis (B*N, F).
            for key in _GRAPH_NODE_KEYS:
                if key in keys:
                    out[key] = torch.cat([b[key] for b in batch], dim=0)

            # Edge indices → offset per sample, then concat along E axis.
            for key in _EDGE_INDEX_KEYS:
                if key in keys:
                    out[key] = _concat_edge_indices(
                        [b[key] for b in batch], n_nodes=n_nodes,
                    )

            # Batch-local stock index within the mega-graph:
            # global_stock_idx = batch_i * N + stock_idx.
            if "stock_idx" in keys:
                offsets = (
                    torch.arange(len(batch), dtype=torch.long) * n_nodes
                )
                out["graph_stock_idx"] = out["stock_idx"] + offsets

            # Record batch size so downstream encoders know to reshape.
            out["_batch_size"] = torch.as_tensor(len(batch), dtype=torch.long)

        # 3. Catch-all: any key not handled above gets default_collate.
        # This keeps the collate function forward-compatible with new
        # optional tensors introduced by downstream experiments.
        unhandled = set(keys) - set(out.keys())
        # ``_batch_size`` is synthetic; never expected in input keys.
        unhandled.discard("_batch_size")
        if unhandled:
            fallback = default_collate(batch)
            for key in unhandled:
                if key in fallback:
                    out[key] = fallback[key]

        return out

    return collate


def _concat_edge_indices(
    tensors: Iterable[torch.Tensor], *, n_nodes: int,
) -> torch.Tensor:
    """Offset and concatenate per-sample edge_index tensors.

    Parameters
    ----------
    tensors
        Per-sample ``(2, E_i)`` edge indices into a single-graph node
        numbering ``[0, N-1]``.
    n_nodes
        Number of nodes per sample. The offset for the ``i``-th sample is
        ``i * n_nodes`` so the batched graph has ``B * n_nodes`` nodes
        and disjoint edge sets.

    Returns
    -------
    torch.Tensor
        ``(2, sum_i E_i)`` long tensor.
    """
    offset_tensors: list[torch.Tensor] = []
    for i, edge_index in enumerate(tensors):
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError(
                f"edge_index must be shape (2, E); got {tuple(edge_index.shape)}"
            )
        offset_tensors.append(edge_index + i * n_nodes)
    return torch.cat(offset_tensors, dim=1)


__all__ = ["make_collate_fn"]
