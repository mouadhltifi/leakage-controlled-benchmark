"""HF-model-agnostic article encoder for news text.

Ports the heavy-lifting logic from v1 ``src/features/encode_news.py``
into a clean, model-agnostic interface. Two public functions:

* :func:`encode_articles_sentiments` — FinBERT 3-class classifier path.
  Reproduces ``news_per_article_sentiments.parquet``.
* :func:`encode_articles_embeddings` — generic HF-model CLS or
  last-token pooling that writes ``emb_*`` columns. Used for FinBERT
  768-dim as well as BGE/FinLang/Qwen3 variants.

The platform does NOT call these during Milestone 3 — the cached
parquets on disk are the primary path. These functions exist so that:

* ``news_per_article_sentiments.parquet`` can be rebuilt if corrupted.
* A *new* encoder (DeBERTa-v3-financial, say) can be screened without
  touching v1 code.

Design notes
------------

* ``transformers``/``torch`` are imported lazily inside the functions so
  the rest of the package stays HF-free for cheap imports (e.g. when
  running unit tests that don't need encoding).
* Test coverage skips these functions unless the env var
  ``RUN_ENCODING_TESTS`` is set. The functions themselves are
  single-purpose and well-trodden — the risk is downloading ~500 MB of
  weights during CI.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

#: Columns returned by :func:`encode_articles_sentiments`.
SENTIMENT_COLUMNS: list[str] = ["Date", "Ticker", "p_pos", "p_neg", "p_neu"]

Pooling = Literal["cls", "mean", "last_token"]


def _clean_text_batch(batch: list[object], max_char_len: int = 512) -> list[str]:
    """Robustly coerce a list of text inputs to clean strings.

    * ``NaN`` / ``None`` -> ``""`` (empty string; tokenisers accept these).
    * Other objects -> ``str(x)``, truncated to ``max_char_len`` chars.
    """
    out: list[str] = []
    for t in batch:
        if t is None or (isinstance(t, float) and not np.isfinite(t)):
            out.append("")
            continue
        if isinstance(t, float) and pd.isna(t):
            out.append("")
            continue
        out.append(str(t)[:max_char_len])
    return out


def _select_device(device: str) -> str:
    """Resolve device strings with MPS/CUDA fallback to CPU.

    Isolated here so the encoder helpers don't depend on
    :mod:`mmfp.utils.device` directly (they're rarely called and their
    dependency on ``torch`` is already explicit).
    """
    import torch

    if device == "auto":
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        log.warning("MPS requested but unavailable; falling back to CPU")
        return "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA requested but unavailable; falling back to CPU")
        return "cpu"
    return device


def encode_articles_sentiments(
    articles: pd.DataFrame,
    *,
    model_name: str = "ProsusAI/finbert",
    batch_size: int = 32,
    device: str = "auto",
    text_column: str = "Title",
    max_length: int = 128,
) -> pd.DataFrame:
    """Encode FinBERT 3-class sentiment probabilities per article.

    Parameters
    ----------
    articles
        Tidy per-article ``DataFrame`` containing at minimum:

        * ``Date`` — ``datetime``, tz-aware or naive.
        * ``Ticker`` — canonical study symbol.
        * ``text_column`` — text to encode (headline by default).
    model_name
        HuggingFace model ID for a 3-class ``AutoModelForSequenceClassification``.
        Defaults to the study's standard FinBERT checkpoint.
    batch_size
        Number of articles per forward pass. 32 is safe on an
        8 GB MPS device for FinBERT.
    device
        ``"mps"``, ``"cuda"``, ``"cpu"``, or ``"auto"``.
    text_column
        Column containing text to encode.
    max_length
        Token truncation ceiling. 128 matches v1 and captures most
        headlines.

    Returns
    -------
    pandas.DataFrame
        Columns ``[Date, Ticker, p_pos, p_neg, p_neu]``. Index is a
        fresh ``RangeIndex``. Row order matches ``articles``.

    Raises
    ------
    ValueError
        If ``text_column`` is missing or ``articles`` is empty.

    Notes
    -----
    FinBERT label order from the HF config is
    ``{0: "positive", 1: "negative", 2: "neutral"}``. We trust the
    config rather than hard-coding the order so a retrained checkpoint
    would still work.
    """
    if articles.empty:
        raise ValueError("encode_articles_sentiments: articles is empty")
    for required in ("Date", "Ticker", text_column):
        if required not in articles.columns:
            raise ValueError(
                f"articles DataFrame missing required column {required!r}"
            )

    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

    resolved_device = _select_device(device)

    log.info(
        "Loading sentiment model %s on %s (%d articles)",
        model_name,
        resolved_device,
        len(articles),
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model = model.to(resolved_device)
    # Switch to inference mode (disables dropout, batchnorm updates, etc.).
    model.eval()

    # Trust the HF config for label positions.
    id2label = getattr(model.config, "id2label", {0: "positive", 1: "negative", 2: "neutral"})
    # Reverse lookup: label_name -> column index.
    label_to_index = {
        "positive": None,
        "negative": None,
        "neutral": None,
    }
    for idx, label in id2label.items():
        norm = str(label).lower()
        if norm in label_to_index:
            label_to_index[norm] = int(idx)
    if any(v is None for v in label_to_index.values()):
        raise ValueError(
            f"Model {model_name!r} does not expose the 3 FinBERT labels "
            f"(positive/negative/neutral). Got id2label={id2label!r}."
        )

    texts = articles[text_column].tolist()
    all_probs: list[np.ndarray] = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = _clean_text_batch(texts[i : i + batch_size])
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(resolved_device)
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)

    probs = np.concatenate(all_probs, axis=0)
    p_pos = probs[:, label_to_index["positive"]]
    p_neg = probs[:, label_to_index["negative"]]
    p_neu = probs[:, label_to_index["neutral"]]

    out = pd.DataFrame(
        {
            "Date": articles["Date"].to_numpy(),
            "Ticker": articles["Ticker"].to_numpy(),
            "p_pos": p_pos.astype(np.float32),
            "p_neg": p_neg.astype(np.float32),
            "p_neu": p_neu.astype(np.float32),
        }
    )
    return out


def encode_articles_embeddings(
    articles: pd.DataFrame,
    *,
    model_name: str,
    batch_size: int = 32,
    device: str = "auto",
    text_column: str = "Title",
    max_length: int = 128,
    pooling: Pooling = "cls",
    l2_normalize: bool = True,
) -> pd.DataFrame:
    """Encode articles into a ``D``-dim embedding per row.

    Works for any HuggingFace model compatible with ``AutoModel``
    (encoder-only or decoder-only). ``last_token`` pooling is required
    for decoder models (Qwen3 Embedding) where the useful
    representation lives in the final token.

    Parameters
    ----------
    articles
        Tidy per-article ``DataFrame`` with at least ``Date``,
        ``Ticker``, ``text_column`` columns.
    model_name
        HF model ID. Examples: ``"ProsusAI/finbert"``,
        ``"BAAI/bge-base-en-v1.5"``, ``"Qwen/Qwen3-Embedding-0.6B"``.
    batch_size
        Articles per forward pass. Larger embeddings (e.g. Qwen3's
        1024-dim) benefit from smaller batches.
    device
        ``"mps"``, ``"cuda"``, ``"cpu"``, or ``"auto"``.
    text_column
        Source column for text.
    max_length
        Token truncation ceiling.
    pooling
        ``"cls"`` for BERT-family models, ``"mean"`` for mean-pool
        across tokens, ``"last_token"`` for decoder models.
    l2_normalize
        When ``True``, return unit-norm rows. Matches the v1 behaviour
        for BGE/Qwen3 and is the input distribution every downstream
        aggregation strategy was developed against.

    Returns
    -------
    pandas.DataFrame
        Columns ``[Date, Ticker, emb_0, emb_1, ..., emb_{D-1}]``.
    """
    if articles.empty:
        raise ValueError("encode_articles_embeddings: articles is empty")
    for required in ("Date", "Ticker", text_column):
        if required not in articles.columns:
            raise ValueError(
                f"articles DataFrame missing required column {required!r}"
            )
    if pooling not in ("cls", "mean", "last_token"):
        raise ValueError(
            f"pooling must be one of 'cls', 'mean', 'last_token'; got {pooling!r}"
        )

    import torch
    from transformers import AutoModel, AutoTokenizer

    resolved_device = _select_device(device)
    log.info(
        "Loading embedding model %s on %s (%d articles, pooling=%s)",
        model_name,
        resolved_device,
        len(articles),
        pooling,
    )

    # Decoder models like Qwen3 need left-padding so the last token
    # position is consistent across the batch.
    padding_side = "left" if pooling == "last_token" else "right"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = padding_side
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModel.from_pretrained(model_name)
    model = model.to(resolved_device)
    # Switch to inference mode (disables dropout, batchnorm updates, etc.).
    model.eval()

    texts = articles[text_column].tolist()
    all_emb: list[np.ndarray] = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = _clean_text_batch(texts[i : i + batch_size])
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(resolved_device)
            outputs = model(**inputs)
            hidden = outputs.last_hidden_state  # (B, T, D)

            if pooling == "cls":
                emb = hidden[:, 0, :]
            elif pooling == "last_token":
                emb = hidden[:, -1, :]
            else:  # mean
                attn = inputs.get("attention_mask")
                if attn is None:
                    emb = hidden.mean(dim=1)
                else:
                    mask = attn.unsqueeze(-1).float()
                    summed = (hidden * mask).sum(dim=1)
                    denom = mask.sum(dim=1).clamp(min=1.0)
                    emb = summed / denom

            if l2_normalize:
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)

            all_emb.append(emb.cpu().numpy())

    emb_matrix = np.concatenate(all_emb, axis=0)
    d_model = emb_matrix.shape[1]
    emb_cols = [f"emb_{i}" for i in range(d_model)]

    out = pd.DataFrame(emb_matrix, columns=emb_cols, dtype=np.float32)
    out.insert(0, "Ticker", articles["Ticker"].to_numpy())
    out.insert(0, "Date", articles["Date"].to_numpy())
    return out


__all__ = [
    "Pooling",
    "SENTIMENT_COLUMNS",
    "encode_articles_embeddings",
    "encode_articles_sentiments",
]
