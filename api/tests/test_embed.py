"""Tests for Task 6.1 — embeddings with a content-addressed cache.

The fake encoder stands in for the real model (§11.2): the cache behaviour,
batching contract and dim-safety checks are all exercised offline.
"""

from __future__ import annotations

import numpy as np
import pytest

import store
from config import Settings
from conftest import make_paper
from pipeline.embed import (
    check_embedding_dims,
    embed_papers,
    embedding_matrix,
    embedding_text,
    get_embedder,
)


class FakeEmbedder:
    def __init__(self, dim: int = 8, model_name: str = "fake-embed") -> None:
        self._dim = dim
        self._model_name = model_name
        self.encode_calls = 0
        self.seen: list[list[str]] = []

    @property
    def model_name(self) -> str:
        return self._model_name

    def encode(self, texts) -> np.ndarray:
        self.encode_calls += 1
        self.seen.append(list(texts))
        rng = np.random.default_rng(0)
        return rng.normal(size=(len(texts), self._dim)).astype(np.float32)


def test_embedding_text_includes_title_and_abstract():
    paper = make_paper(title="T", abstract="A")
    assert embedding_text(paper) == "T\nA"


def test_get_embedder_honours_device_from_settings(settings: Settings):
    from dataclasses import replace

    cpu = get_embedder(replace(settings, embed_device="cpu"))
    assert cpu.device == "cpu"
    other = get_embedder(replace(settings, embed_device="mps"))
    assert other.device == "mps"


def test_embeddings_are_cached_across_calls(settings: Settings, conn):
    papers = [make_paper(paper_id="p1"), make_paper(paper_id="p2")]
    store.upsert_papers(conn, papers)
    embedder = FakeEmbedder()

    first = embed_papers(papers, settings, conn, embedder=embedder)
    assert embedder.encode_calls == 1

    second = embed_papers(papers, settings, conn, embedder=embedder)
    assert embedder.encode_calls == 1  # served from the cache
    assert np.allclose(first["p1"], second["p1"])


def test_only_new_papers_are_encoded(settings: Settings, conn):
    """Growing a landscape must not re-embed what it already knows."""
    embedder = FakeEmbedder()
    existing = [make_paper(paper_id=f"p{i}") for i in range(3)]
    store.upsert_papers(conn, existing)
    embed_papers(existing, settings, conn, embedder=embedder)
    assert embedder.encode_calls == 1

    newcomer = make_paper(paper_id="new")
    store.upsert_papers(conn, [newcomer])
    vectors = embed_papers([*existing, newcomer], settings, conn, embedder=embedder)

    assert embedder.encode_calls == 2
    assert set(vectors) == {"p0", "p1", "p2", "new"}


def test_embeddings_roundtrip_through_the_store(settings: Settings, conn):
    papers = [make_paper(paper_id="p1")]
    store.upsert_papers(conn, papers)
    embedder = FakeEmbedder(dim=4)
    expected = embed_papers(papers, settings, conn, embedder=embedder)["p1"]

    fetched = store.fetch_embeddings(conn, ["p1"], embedder.model_name)["p1"]
    restored = np.frombuffer(fetched, dtype=np.float32)
    assert np.allclose(restored, expected)


def test_embeddings_are_scoped_by_model(settings: Settings, conn):
    papers = [make_paper(paper_id="p1")]
    store.upsert_papers(conn, papers)
    embed_papers(papers, settings, conn, embedder=FakeEmbedder(model_name="model-a"))
    other = FakeEmbedder(model_name="model-b")
    embed_papers(papers, settings, conn, embedder=other)
    # A different model has no cache, so it must actually encode.
    assert other.encode_calls == 1


def test_dim_mismatch_names_both_dims(settings: Settings, conn):
    """Swapping embedders under a warm cache fails loudly, not silently."""
    papers = [make_paper(paper_id="p1")]
    store.upsert_papers(conn, papers)
    embed_papers(papers, settings, conn, embedder=FakeEmbedder(dim=8))

    with pytest.raises(ValueError, match="cached dim 8.*freshly encoded dim 4"):
        embed_papers(papers, settings, conn, embedder=FakeEmbedder(dim=4))


def test_dim_check_against_raw_blobs():
    cached = {"p1": np.zeros(8, dtype=np.float32).tobytes()}
    with pytest.raises(ValueError, match="cached dim 8.*freshly encoded dim 4"):
        check_embedding_dims(cached, 4, "fake-embed")
    check_embedding_dims(cached, 8, "fake-embed")  # matching dims pass silently


def test_embedding_of_nothing_is_empty(settings: Settings):
    assert embed_papers([], settings, None, embedder=FakeEmbedder()) == {}


def test_a_mismatched_vector_count_is_an_error(settings: Settings):
    class ShortEmbedder(FakeEmbedder):
        def encode(self, texts):
            self.encode_calls += 1
            return np.zeros((1, 4), dtype=np.float32)  # always one row

    with pytest.raises(ValueError, match="vectors for"):
        embed_papers(
            [make_paper(paper_id="a"), make_paper(paper_id="b")],
            settings,
            None,
            embedder=ShortEmbedder(),
        )


def test_embedding_matrix_preserves_the_requested_order():
    vectors = {
        "b": np.array([2.0, 0.0], dtype=np.float32),
        "a": np.array([1.0, 0.0], dtype=np.float32),
    }
    matrix = embedding_matrix(["a", "b"], vectors)
    assert matrix[0][0] == pytest.approx(1.0)
    assert matrix[1][0] == pytest.approx(2.0)


def test_embedding_matrix_refuses_to_silently_drop_a_missing_paper():
    """Dropping a row would shift every later coordinate onto the wrong paper."""
    with pytest.raises(ValueError, match="Missing embeddings"):
        embedding_matrix(["a", "missing"], {"a": np.zeros(3, dtype=np.float32)})


def test_embedding_matrix_of_nothing_is_empty():
    assert embedding_matrix([], {}).shape == (0, 0)
