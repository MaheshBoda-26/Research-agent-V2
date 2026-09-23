"""Tests for Task 7.2 — citation edges projected onto the corpus.

Pinned behaviours (plan Task 7.2):

* intra-corpus citation links appear with the correct direction (src cites dst);
* cross-corpus links are dropped (both ends must be in the corpus);
* when a kNN and a citation edge cover the same pair, **one** edge survives,
  marked ``source='citation'`` (a citation is stronger evidence than proximity);
* a paper cited by 30 in-corpus papers does not produce 30 edges after the cap
  (Task 7.3's :func:`cap_edges_per_node`).
"""

from __future__ import annotations

import numpy as np

from pipeline.graph import (
    cap_edges_per_node,
    citation_edges,
    knn_edges,
    merge_edges,
)


def _edge(src: str, dst: str, source: str, confidence: float) -> dict:
    return {
        "src_paper_id": src,
        "dst_paper_id": dst,
        "kind": "applies",
        "weight": 0.5,
        "rationale": "",
        "source": source,
        "confidence": confidence,
        "how": source,
        "context": "",
    }


def test_intra_corpus_citations_keep_direction():
    triples = [("a", "b", "openalex"), ("c", "a", "semanticscholar")]
    edges = citation_edges(triples, {"a", "b", "c"})
    assert len(edges) == 2
    by_src = {e["src_paper_id"]: e for e in edges}
    assert by_src["a"]["dst_paper_id"] == "b"
    assert by_src["c"]["dst_paper_id"] == "a"
    for e in edges:
        assert e["source"] == "citation"
        assert e["confidence"] == 1.0


def test_cross_corpus_links_are_dropped():
    triples = [
        ("a", "b", "openalex"),      # both in corpus
        ("a", "x", "openalex"),      # dst outside
        ("y", "a", "openalex"),      # src outside
        ("x", "y", "openalex"),      # both outside
    ]
    edges = citation_edges(triples, {"a", "b"})
    assert len(edges) == 1
    assert edges[0]["src_paper_id"] == "a"
    assert edges[0]["dst_paper_id"] == "b"


def test_self_citation_is_dropped():
    edges = citation_edges([("a", "a", "openalex")], {"a"})
    assert edges == []


def test_citation_beats_knn_on_the_same_pair():
    """The merged layer keeps a single citation edge for a shared pair."""
    # kNN sees a-b at cosine 0.8; a citation a->b also exists.
    knn = [_edge("a", "b", "knn", 0.8)]
    cit = [_edge("a", "b", "citation", 1.0)]
    merged = merge_edges([knn, cit])
    assert len(merged) == 1
    assert merged[0]["source"] == "citation"
    assert merged[0]["confidence"] == 1.0


def test_merge_dedupes_undirected_pairs_within_a_layer():
    """A->B and B->A in the same layer collapse; higher confidence wins."""
    knn = [
        _edge("a", "b", "knn", 0.75),
        _edge("b", "a", "knn", 0.90),
    ]
    merged = merge_edges([knn])
    assert len(merged) == 1
    assert merged[0]["confidence"] == 0.90


def test_hub_paper_is_capped():
    """A paper cited by 30 in-corpus papers yields at most `cap` edges (7.3)."""
    hub = "hub"
    spokes = [f"s{i}" for i in range(30)]
    triples = [(s, hub, "openalex") for s in spokes]
    edges = citation_edges(triples, {hub, *spokes})
    assert len(edges) == 30  # the citation layer itself is complete...
    capped = cap_edges_per_node(edges, cap=6)
    # ...but after the cap the hub has at most 6 edges.
    hub_degree = sum(
        1 for e in capped if hub in (e["src_paper_id"], e["dst_paper_id"])
    )
    assert hub_degree <= 6


def test_citation_layer_via_knn_path_still_marks_provenance():
    """kNN edges built from real embeddings merge correctly with citations."""
    ids = ["a", "b"]
    matrix = np.eye(2) * 1.0  # orthogonal — no kNN edges
    knn = knn_edges(ids, matrix, k=1, threshold=0.72)
    assert knn == []
    cit = citation_edges([("a", "b", "openalex")], {"a", "b"})
    merged = merge_edges([knn, cit])
    assert len(merged) == 1
    assert merged[0]["source"] == "citation"
