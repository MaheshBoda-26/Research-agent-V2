"""Embeddings, cached per (paper, model).

Vectors are L2-normalized so cosine similarity is a dot product, and stored as
raw float32 bytes in ``paper_embeddings``. That cache is what lets a landscape
grow without re-embedding anything it already knows about: adding 20 papers to a
200-paper topic costs 20 encodes, not 220.

The default model is ``BAAI/bge-base-en-v1.5`` — a general-purpose encoder that
is small, fast on CPU, and strong on short technical passages. It is not a
scientific-document model; SPECTER2 would be the specialist choice but needs the
extra ``adapters`` package, so it stays behind the same ``EMBED_MODEL`` knob as
an experiment rather than the default.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np

from config import Settings
from models import Paper

logger = logging.getLogger(__name__)

DEFAULT_EMBED_MODEL = "BAAI/bge-base-en-v1.5"


class EmbedderLike(Protocol):
    """Minimal surface, so tests never download a model."""

    @property
    def model_name(self) -> str: ...

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class Embedder:
    """Lazy wrapper around a sentence-transformers encoder."""

    def __init__(self, model_name: str = DEFAULT_EMBED_MODEL, device: str = "cpu") -> None:
        self._model_name = model_name
        self.device = device
        self._model: Any = None

    @property
    def model_name(self) -> str:
        return self._model_name

    def _load(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name, device=self.device)
            logger.info("Loaded embedding model %s on %s", self._model_name, self.device)
        return self._model

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return an (n, dim) float32 matrix of L2-normalized vectors."""
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        model = self._load()
        vectors = model.encode(
            list(texts),
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(vectors, dtype=np.float32)


#: Process-wide default embedder, so the model loads once per server.
_embedder: Embedder | None = None


def get_embedder(settings: Settings) -> Embedder:
    """Return the shared embedder, honouring model *and* device from settings."""
    global _embedder
    if (
        _embedder is None
        or _embedder.model_name != settings.embed_model
        or _embedder.device != settings.embed_device
    ):
        _embedder = Embedder(settings.embed_model, device=settings.embed_device)
    return _embedder


def embedding_text(paper: Paper) -> str:
    """Text that gets embedded: title first, because it carries the topic."""
    return f"{paper.title}\n{paper.abstract}"


def check_embedding_dims(cached: dict[str, bytes], fresh_dim: int, model: str) -> None:
    """Hard check that cached vectors agree with freshly encoded ones.

    Swapping embedders under a warm cache would otherwise write misaligned
    vectors side by side — every cosine after that is garbage. The stored
    ``dim`` column is authoritative, but the fetch surface returns bytes, so
    the check compares byte lengths (4 bytes per float32): any cached vector
    whose length disagrees with the fresh dim fails loudly, naming both dims.
    """
    for paper_id, blob in cached.items():
        cached_dim = len(blob) // 4
        if cached_dim != fresh_dim:
            raise ValueError(
                f"Embedding dim mismatch for model {model!r} on paper {paper_id!r}: "
                f"cached dim {cached_dim} != freshly encoded dim {fresh_dim}. "
                "The embedder was swapped without clearing paper_embeddings."
            )


def embed_papers(
    papers: Sequence[Paper],
    settings: Settings,
    conn: sqlite3.Connection | None = None,
    *,
    embedder: EmbedderLike | None = None,
) -> dict[str, np.ndarray]:
    """Return ``paper_id -> float32 vector``, encoding only what is not cached."""
    if not papers:
        return {}

    active = embedder if embedder is not None else get_embedder(settings)
    model_name = active.model_name

    cached: dict[str, bytes] = {}
    if conn is not None:
        from store import fetch_embeddings

        cached = fetch_embeddings(conn, [p.paper_id for p in papers], model_name)

    todo = [p for p in papers if p.paper_id not in cached]
    logger.info("Embeddings: %d cached, %d to encode (%s)", len(cached), len(todo), model_name)

    if todo:
        matrix = active.encode([embedding_text(p) for p in todo])
        if matrix.shape[0] != len(todo):
            raise ValueError(
                f"Embedder returned {matrix.shape[0]} vectors for {len(todo)} papers"
            )
        fresh = {
            paper.paper_id: matrix[index].astype(np.float32)
            for index, paper in enumerate(todo)
        }
        if matrix.shape[1:] and matrix.shape[1] > 0:
            check_embedding_dims(cached, int(matrix.shape[1]), model_name)
        if conn is not None:
            from store import upsert_embeddings

            upsert_embeddings(
                conn, model_name, {pid: vec.tobytes() for pid, vec in fresh.items()}
            )
            conn.commit()
    else:
        fresh = {}

    out: dict[str, np.ndarray] = {}
    for paper_id, blob in cached.items():
        out[paper_id] = np.frombuffer(blob, dtype=np.float32).copy()
    out.update(fresh)
    return out


def embedding_matrix(
    paper_ids: Sequence[str], vectors: dict[str, np.ndarray]
) -> np.ndarray:
    """Stack vectors in ``paper_ids`` order, failing loudly on a missing one.

    Order matters: the matrix rows must line up with the id list that the layout
    and the UI both index by. Silently dropping a missing row would shift every
    subsequent coordinate onto the wrong paper.
    """
    missing = [pid for pid in paper_ids if pid not in vectors]
    if missing:
        raise ValueError(f"Missing embeddings for {len(missing)} paper(s): {missing[:5]}")
    if not paper_ids:
        return np.zeros((0, 0), dtype=np.float32)
    return np.vstack([vectors[pid] for pid in paper_ids]).astype(np.float32)


__all__ = [
    "DEFAULT_EMBED_MODEL",
    "Embedder",
    "EmbedderLike",
    "check_embedding_dims",
    "embed_papers",
    "embedding_matrix",
    "embedding_text",
    "get_embedder",
]
