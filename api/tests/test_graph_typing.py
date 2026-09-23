"""Tests for Task 7.3 — cap, density floor, and typology classification.

Pinned behaviours (plan Task 7.3):

* the LLM **may not invent edges** — a response containing an un-supplied pair
  is dropped;
* ``unknown`` falls back deterministically (``shares_method`` for kNN,
  ``applies`` for citation);
* a response that omits a candidate leaves that edge's default intact;
* the per-node cap keeps the strongest edges (citation confidence 1.0 > cosine);
* the density floor lowers the cosine threshold in 0.05 steps until
  ``graph_min_density`` is met or ``graph_cosine_floor`` is reached.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from config import Settings
from models import EdgeTyping, TypedEdge
from pipeline.graph import (
    cap_edges_per_node,
    classify_edges,
    edge_density,
    enforce_density_floor,
    typing_default_for,
)


class TypingCompleter:
    """Hands back a canned EdgeTyping envelope (offline JSONCompleter, §11.2).

    ``raw`` lets a test return an unvalidated dict (e.g. containing a kind of
    ``"unknown"``) to exercise the lenient per-edge parser.
    """

    def __init__(
        self,
        edges: list[TypedEdge] | None,
        *,
        raises: bool = False,
        raw: dict | None = None,
    ):
        self.envelope = EdgeTyping(edges=edges or [])
        self.raw = raw
        self.raises = raises
        self.calls: list[dict] = []

    def complete_json(self, *, system, user, schema, stage="", **kwargs):
        self.calls.append({"system": system, "user": user, "stage": stage})
        if self.raises:
            raise RuntimeError("model unavailable")
        return self.raw if self.raw is not None else self.envelope

    def complete_text(self, *, system, user):
        return None


def _candidate(src: str, dst: str, source: str, confidence: float = 0.8) -> dict:
    return {
        "src_paper_id": src,
        "dst_paper_id": dst,
        "kind": "shares_method" if source == "knn" else "applies",
        "weight": 0.5,
        "rationale": "",
        "source": source,
        "confidence": confidence,
        "how": "kNN" if source == "knn" else "citation",
        "context": "",
    }


def _edge(src: str, dst: str, confidence: float = 0.8, source: str = "knn") -> dict:
    return {
        "src_paper_id": src,
        "dst_paper_id": dst,
        "kind": "shares_method",
        "weight": 0.5,
        "rationale": "",
        "source": source,
        "confidence": confidence,
        "how": "kNN",
        "context": "",
    }


# --------------------------------------------------------------------------- #
# Typology classification
# --------------------------------------------------------------------------- #


def test_invented_pair_is_dropped():
    """An edge the LLM returns that was never supplied must not appear."""
    candidates = [_candidate("a", "b", "knn")]
    completer = TypingCompleter(
        [
            TypedEdge(src_paper_id="a", dst_paper_id="b", kind="extends"),
            TypedEdge(src_paper_id="a", dst_paper_id="zzz", kind="contradicts"),
        ]
    )
    out = classify_edges(candidates, completer, "topic", {"a", "b"}, {})
    assert len(out) == 1
    assert (out[0]["src_paper_id"], out[0]["dst_paper_id"]) == ("a", "b")
    assert out[0]["kind"] == "extends"
    assert not any(e["dst_paper_id"] == "zzz" for e in out)


def test_unknown_falls_back_to_deterministic_default():
    candidates = [
        _candidate("a", "b", "knn"),
        _candidate("c", "d", "citation"),
    ]
    # A raw payload (as a lenient provider or repair-loop survivor may yield)
    # carrying kind="unknown" — never persisted, mapped to the default per edge.
    completer = TypingCompleter(
        None,
        raw={
            "edges": [
                {"src_paper_id": "a", "dst_paper_id": "b", "kind": "unknown"},
                {"src_paper_id": "c", "dst_paper_id": "d", "kind": "unknown"},
            ]
        },
    )
    out = classify_edges(candidates, completer, "topic", {"a", "b", "c", "d"}, {})
    by_pair = {(e["src_paper_id"], e["dst_paper_id"]): e for e in out}
    assert by_pair[("a", "b")]["kind"] == "shares_method"  # knn default
    assert by_pair[("c", "d")]["kind"] == "applies"        # citation default


def test_omitted_candidate_keeps_default():
    candidates = [_candidate("a", "b", "knn"), _candidate("c", "d", "knn")]
    completer = TypingCompleter(
        [TypedEdge(src_paper_id="a", dst_paper_id="b", kind="contradicts")]
    )
    out = classify_edges(candidates, completer, "topic", {"a", "b", "c", "d"}, {})
    by_pair = {(e["src_paper_id"], e["dst_paper_id"]): e for e in out}
    assert by_pair[("a", "b")]["kind"] == "contradicts"      # labelled
    assert by_pair[("c", "d")]["kind"] == "shares_method"    # default intact


def test_failed_call_keeps_all_deterministic_defaults():
    candidates = [_candidate("a", "b", "knn"), _candidate("c", "d", "citation")]
    completer = TypingCompleter(None, raises=True)
    out = classify_edges(candidates, completer, "topic", {"a", "b", "c", "d"}, {})
    by_pair = {(e["src_paper_id"], e["dst_paper_id"]): e for e in out}
    assert by_pair[("a", "b")]["kind"] == "shares_method"
    assert by_pair[("c", "d")]["kind"] == "applies"


def test_internal_fields_are_stripped():
    candidates = [_candidate("a", "b", "knn")]
    completer = TypingCompleter(
        [TypedEdge(src_paper_id="a", dst_paper_id="b", kind="extends")]
    )
    out = classify_edges(candidates, completer, "topic", {"a", "b"}, {})
    assert "how" not in out[0]
    assert "context" not in out[0]


def test_empty_candidates_short_circuit():
    completer = TypingCompleter([])
    assert classify_edges([], completer, "topic", set(), {}) == []
    assert completer.calls == []  # no LLM call for an empty candidate list


def test_typing_default_for_mapping():
    assert typing_default_for("knn") == "shares_method"
    assert typing_default_for("citation") == "applies"
    assert typing_default_for("anything-else") == "shares_method"


# --------------------------------------------------------------------------- #
# Cap per node
# --------------------------------------------------------------------------- #


def test_cap_keeps_strongest_edges():
    hub = "h"
    confidences = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65]
    edges = [_edge(hub, f"n{i}", c) for i, c in enumerate(confidences)]
    capped = cap_edges_per_node(edges, cap=3)
    assert len(capped) == 3
    assert sorted((e["confidence"] for e in capped), reverse=True) == [0.95, 0.90, 0.85]


def test_cap_respects_both_endpoints():
    # a-b(0.9) is kept; b-c(0.8) is then dropped because b has reached the cap.
    edges = [_edge("a", "b", 0.9), _edge("b", "c", 0.8)]
    capped = cap_edges_per_node(edges, cap=1)
    assert len(capped) == 1
    assert (capped[0]["src_paper_id"], capped[0]["dst_paper_id"]) == ("a", "b")
    assert capped[0]["confidence"] == 0.9


def test_cap_citation_beats_cosine():
    """Citation edges (confidence 1.0) survive the cap ahead of kNN edges."""
    hub = "h"
    edges = [_edge(hub, "cit", 1.0, source="citation")]
    edges += [_edge(hub, f"k{i}", 0.99) for i in range(6)]
    capped = cap_edges_per_node(edges, cap=6)
    sources = {(e["src_paper_id"], e["dst_paper_id"]): e["source"] for e in capped}
    assert sources[("h", "cit")] == "citation"


# --------------------------------------------------------------------------- #
# Density floor
# --------------------------------------------------------------------------- #


def test_edge_density_metric():
    edges = [_edge("a", "b"), _edge("b", "c")]
    assert edge_density(edges, 3) == 2 / 3
    assert edge_density([], 0) == 0.0


def test_density_floor_lowers_threshold_until_met(settings: Settings):
    """A corpus whose pairs sit below the 0.72 start needs a lowered threshold."""
    rng = np.random.default_rng(3)
    base = np.zeros(8)
    base[0] = 1.0
    vecs = [base + rng.normal(scale=0.4, size=8) for _ in range(4)]
    matrix = np.vstack(vecs)
    matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    ids = [f"p{i}" for i in range(4)]

    # Start above the data's cosine; loosen the density target so a connected
    # 4-node graph can satisfy it once the threshold drops.
    tuned = replace(
        settings,
        graph_min_density=1.0,
        graph_cosine_start=0.95,
        graph_cosine_floor=0.50,
    )
    edges, threshold_used = enforce_density_floor(ids, matrix, [], tuned)
    assert threshold_used < 0.95  # the threshold had to drop
    assert edges                  # and edges were captured once it did


def test_density_floor_never_goes_below_floor(settings: Settings):
    """An orthogonal corpus can never reach the density floor — stop at 0.60."""
    ids = [f"p{i}" for i in range(6)]
    matrix = np.eye(6)
    edges, threshold_used = enforce_density_floor(ids, matrix, [], settings)
    assert threshold_used == settings.graph_cosine_floor
    assert edges == []  # orthogonal: no edges even at the floor


def test_density_floor_empty_corpus(settings: Settings):
    edges, threshold_used = enforce_density_floor([], np.zeros((0, 4)), [], settings)
    assert edges == []
    assert threshold_used == settings.graph_cosine_start

