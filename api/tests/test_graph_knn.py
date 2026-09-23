"""Tests for Task 7.1 — deterministic semantic kNN edges (D2 islands fix).

Pinned behaviours (plan Task 7.1):

* a synthetic corpus of 3 tight groups yields >=1 edge within each group and
  <=1 edge between groups;
* an all-orthogonal corpus yields **0** edges rather than a forced graph;
* every edge carries ``source='knn'`` and ``confidence=cosine``;
* undirected pairs are deduped (A->B and B->A collapse to one edge).

These tests are offline and exact — embeddings are constructed directly, no
embedder or network is involved (plan §11.2).
"""

from __future__ import annotations

import numpy as np

from pipeline.graph import knn_edges


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / norms


def _tight_groups() -> tuple[list[str], np.ndarray]:
    """Three tight clusters around orthogonal centroids, lightly perturbed."""
    rng = np.random.default_rng(7)
    centroids = np.eye(3, 12)  # e0, e1, e2 in R^12
    ids: list[str] = []
    vecs: list[np.ndarray] = []
    for group in range(3):
        for member in range(4):
            ids.append(f"g{group}p{member}")
            jitter = rng.normal(scale=0.05, size=12)
            vecs.append(centroids[group] + jitter)
    return ids, _normalize(np.vstack(vecs))


def test_three_tight_groups_have_internal_edges():
    ids, matrix = _tight_groups()
    edges = knn_edges(ids, matrix, k=3, threshold=0.72)

    def group_of(pid: str) -> int:
        return int(pid[1])

    within = [e for e in edges if group_of(e["src_paper_id"]) == group_of(e["dst_paper_id"])]
    between = [e for e in edges if group_of(e["src_paper_id"]) != group_of(e["dst_paper_id"])]

    # >= 1 edge within each of the three groups.
    for group in range(3):
        assert any(
            group_of(e["src_paper_id"]) == group for e in within
        ), f"group {group} has no internal edge"
    # Tight groups 0.72-apart: few (ideally no) cross-group edges.
    assert len(between) <= 1


def test_orthogonal_corpus_yields_no_edges():
    """Distinct basis vectors have cosine 0 < 0.72 — no edges, not a forced graph."""
    n = 8
    ids = [f"p{i}" for i in range(n)]
    matrix = np.eye(n)  # already L2-normalized, mutually orthogonal
    edges = knn_edges(ids, matrix, k=3, threshold=0.72)
    assert edges == []


def test_edges_carry_knn_provenance_and_cosine_confidence():
    ids, matrix = _tight_groups()
    edges = knn_edges(ids, matrix, k=3, threshold=0.60)
    assert edges  # sanity
    for edge in edges:
        assert edge["source"] == "knn"
        assert 0.0 < edge["confidence"] <= 1.0
        # confidence is the actual cosine similarity of the pair.
        i, j = ids.index(edge["src_paper_id"]), ids.index(edge["dst_paper_id"])
        assert edge["confidence"] == round(float(matrix[i] @ matrix[j]), 4)


def test_undirected_pairs_are_deduped():
    ids, matrix = _tight_groups()
    edges = knn_edges(ids, matrix, k=3, threshold=0.60)
    pairs = [frozenset((e["src_paper_id"], e["dst_paper_id"])) for e in edges]
    assert len(pairs) == len(set(pairs))


def test_threshold_filters_weak_edges():
    ids, matrix = _tight_groups()
    strict = knn_edges(ids, matrix, k=3, threshold=0.99)
    loose = knn_edges(ids, matrix, k=3, threshold=0.50)
    assert len(strict) <= len(loose)
    assert all(e["confidence"] >= 0.99 for e in strict)


def test_empty_and_mismatched_inputs_yield_no_edges():
    assert knn_edges([], np.zeros((0, 4)), k=3, threshold=0.72) == []
    matrix = np.eye(3)
    assert knn_edges(["a", "b"], matrix, k=3, threshold=0.72) == []
