"""Stage 5 — relationship graph (Phase 7, Tasks 7.1-7.3).

Connectivity is computed **before** classification: kNN and citation layers
produce candidate edges deterministically, then the LLM only assigns a ``kind``
to pairs the deterministic layers already believed were related. The LLM may
not invent edges — an invented edge is a dropped pair, never a new row.

Edge dict shape flowing through this module::

    {
        "src_paper_id": str,
        "dst_paper_id": str,
        "kind": str,            # deterministic default; overwritten by LLM typing
        "weight": float,        # 0.5 (kept for store.replace_edges compatibility)
        "rationale": str,       # "" until the LLM fills it
        "source": str,          # "knn" | "citation"
        "confidence": float,    # cosine for knn, 1.0 for citation
        # internal, stripped before persistence:
        "how": str,             # "kNN" | "citation" — shown in the typing prompt
        "context": str,         # one-line evidence from each paper — prompt only
    }

The ``source`` field tracks **provenance** (D2): how the connection was first
discovered. The ``kind`` field is the LLM's classification of that connection.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import numpy as np

from config import Settings
from llm.protocol import JSONCompleter
from models import EdgeKind, Paper, EdgeTyping, TypedEdge
from prompts.graph import build_edge_typing_prompt, parse_typing_result
from prompts.graph import EDGE_CONTEXT_CHARS, EDGE_TYPING_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

#: Default weight for every edge (plan §3.2: deterministic edges start at 0.5).
EDGE_WEIGHT = 0.5

#: Default kind for kNN-derived edges before LLM typing.
KNN_DEFAULT_KIND: EdgeKind = "shares_method"

#: Default kind for citation-derived edges before LLM typing.
CITATION_DEFAULT_KIND: EdgeKind = "applies"

#: How the prompt phrases the provenance of each candidate.
_KNN_HOW = "kNN (semantic similarity)"
_CITATION_HOW = "citation"

# --------------------------------------------------------------------------- #
# Layer 1 — kNN semantic edges (Task 7.1)
# --------------------------------------------------------------------------- #


def knn_edges(
    paper_ids: list[str], embeddings: np.ndarray, *, k: int, threshold: float
) -> list[dict[str, Any]]:
    """Connect each paper to its ``k`` nearest neighbours by cosine similarity.

    * Embeddings must be L2-normalized (the V2 embedder normalizes them).
    * An edge is kept only when cosine >= ``threshold``.
    * Undirected pairs are deduped: if A->B and B->A both appear, only one
      edge survives (the higher cosine wins).

    Every returned edge carries ``source='knn'`` and ``confidence=cosine``.
    """
    n = len(paper_ids)
    if n == 0 or embeddings is None or embeddings.shape[0] != n:
        return []

    # Cosine similarity = dot product for L2-normalized vectors.
    sims = embeddings @ embeddings.T
    np.fill_diagonal(sims, -1.0)

    seen: set[tuple[str, str]] = set()
    edges: list[dict[str, Any]] = []

    for i in range(n):
        row = sims[i]
        # Top-k neighbours by similarity (descending).
        nn_indices = np.argsort(-row)[:k]
        for j in nn_indices:
            if i == j:
                continue
            cos = float(row[j])
            if cos < threshold:
                continue
            # Undirected dedupe: canonical pair order.
            pair = tuple(sorted((paper_ids[i], paper_ids[j])))
            if pair in seen:
                continue
            seen.add(pair)
            edges.append(
                {
                    "src_paper_id": paper_ids[i],
                    "dst_paper_id": paper_ids[j],
                    "kind": KNN_DEFAULT_KIND,
                    "weight": EDGE_WEIGHT,
                    "rationale": "",
                    "source": "knn",
                    "confidence": round(cos, 4),
                    "how": _KNN_HOW,
                    "context": "",
                }
            )
    return edges


# --------------------------------------------------------------------------- #
# Layer 2 — citation edges (Task 7.2)
# --------------------------------------------------------------------------- #


def citation_edges(
    citation_triples: list[tuple[str, str, str]], corpus_ids: set[str]
) -> list[dict[str, Any]]:
    """Project citation links onto the current corpus.

    ``citation_triples`` is ``(src, dst, source)`` where ``src`` cites ``dst``.
    Only links whose both ends are in ``corpus_ids`` are returned; cross-corpus
    links are dropped (they cannot be drawn on this map).

    Direction follows citation order: ``src`` cites ``dst``.
    ``confidence`` is 1.0 (a citation is stronger evidence than proximity).
    """
    edges: list[dict[str, Any]] = []
    for src, dst, _source in citation_triples:
        if src in corpus_ids and dst in corpus_ids and src != dst:
            edges.append(
                {
                    "src_paper_id": src,
                    "dst_paper_id": dst,
                    "kind": CITATION_DEFAULT_KIND,
                    "weight": EDGE_WEIGHT,
                    "rationale": "",
                    "source": "citation",
                    "confidence": 1.0,
                    "how": _CITATION_HOW,
                    "context": "",
                }
            )
    return edges


# --------------------------------------------------------------------------- #
# Merge and cap (Task 7.3)
# --------------------------------------------------------------------------- #


def merge_edges(layers: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Merge edge layers: citation beats kNN, dedupe undirected pairs.

    ``layers`` is ordered from lowest to highest priority — the **last** layer
    in the list wins when two layers share a pair. Pass
    ``[knn_edges, citation_edges]`` so citations override proximities.

    Within the same layer, when a pair appears twice (once per direction), the
    higher-confidence edge wins.
    """
    by_pair: dict[frozenset, dict[str, Any]] = {}
    for layer in layers:
        for edge in layer:
            key = frozenset({edge["src_paper_id"], edge["dst_paper_id"]})
            existing = by_pair.get(key)
            if existing is None:
                by_pair[key] = edge
            else:
                existing_source = existing["source"]
                new_source = edge["source"]
                # Citation always beats kNN.
                if new_source == "citation" and existing_source != "citation":
                    by_pair[key] = edge
                elif new_source == existing_source:
                    # Same provenance: keep the stronger confidence.
                    if edge["confidence"] > existing["confidence"]:
                        by_pair[key] = edge
                # Otherwise existing (higher priority) stays.
    return list(by_pair.values())


def cap_edges_per_node(edges: list[dict[str, Any]], cap: int) -> list[dict[str, Any]]:
    """Cap the degree of every node at ``cap`` edges.

    Edges are sorted by strength (citation confidence 1.0 > cosine) and added
    greedily: an edge is kept only when both endpoints are still under the cap.
    This guarantees no paper is pinned by dozens of neighbours.
    """
    if not edges or cap <= 0:
        return []
    ordered = sorted(edges, key=lambda e: e["confidence"], reverse=True)
    degrees: dict[str, int] = defaultdict(int)
    result: list[dict[str, Any]] = []
    for edge in ordered:
        src, dst = edge["src_paper_id"], edge["dst_paper_id"]
        if degrees[src] >= cap or degrees[dst] >= cap:
            continue
        result.append(edge)
        degrees[src] += 1
        degrees[dst] += 1
    return result


def edge_density(edges: list[dict[str, Any]], paper_count: int) -> float:
    """Edges per paper — the plan's density metric (>=2.5 over the whole map)."""
    if paper_count == 0:
        return 0.0
    return len(edges) / paper_count


# --------------------------------------------------------------------------- #
# Density floor enforcement (Task 7.3)
# --------------------------------------------------------------------------- #


def enforce_density_floor(
    paper_ids: list[str],
    embeddings: np.ndarray,
    citation_edge_list: list[dict[str, Any]],
    settings: Settings,
) -> tuple[list[dict[str, Any]], float]:
    """Lower the kNN cosine threshold until density >= floor or floor-reached.

    Starting at ``graph_cosine_start`` (0.72), decrease by 0.05 steps until
    ``graph_min_density`` (2.5 edges/paper) is met, never going below
    ``graph_cosine_floor`` (0.60). At each threshold the kNN layer is recomputed
    and re-merged with the citation layer.

    Returns ``(edges, threshold_used)`` so the caller can log the actual threshold.
    """
    threshold = settings.graph_cosine_start
    floor = settings.graph_cosine_floor
    paper_count = len(paper_ids)
    if paper_count == 0:
        return [], threshold

    while True:
        nn = knn_edges(
            paper_ids, embeddings, k=settings.graph_knn_k, threshold=threshold
        )
        merged = merge_edges([nn, citation_edge_list])
        density = edge_density(merged, paper_count)

        if density >= settings.graph_min_density:
            logger.info(
                "Density floor met at cosine threshold %.2f (%.2f edges/paper)",
                threshold,
                density,
            )
            return merged, threshold

        if threshold <= floor:
            logger.warning(
                "Density floor not met even at threshold %.2f; "
                "returning %d edges (%.2f edges/paper)",
                threshold,
                len(merged),
                density,
            )
            return merged, threshold

        # Step down, clamped to the floor.
        threshold = max(round(threshold - 0.05, 2), floor)


# --------------------------------------------------------------------------- #
# LLM edge classification (Task 7.3)
# --------------------------------------------------------------------------- #


def typing_default_for(source: str, cross_cluster: bool = False) -> EdgeKind:
    """Deterministic default when the LLM returns ``unknown`` for a candidate.

    * kNN-derived → ``shares_method``
    * citation-derived → ``applies``
    """
    if source == "citation":
        return "applies"
    return "shares_method"


def _edge_context(
    edge: dict[str, Any], papers_by_id: dict[str, Paper]
) -> str:
    """Build the evidence snippet shown to the LLM for one candidate edge."""

    def snippet(pid: str) -> str:
        paper = papers_by_id.get(pid)
        if paper is None:
            return pid
        title = paper.title.strip()
        abstract = " ".join((paper.abstract or "").split())
        if len(abstract) > EDGE_CONTEXT_CHARS:
            abstract = abstract[: EDGE_CONTEXT_CHARS - 1] + "\u2026"
        return f"[{pid}] {title}. {abstract}".strip()

    return snippet(edge["src_paper_id"]) + "\n    " + snippet(edge["dst_paper_id"])


def classify_edges(
    edges: list[dict[str, Any]],
    completer: JSONCompleter,
    topic: str,
    valid_ids: set[str],
    papers_by_id: dict[str, Paper],
    cluster_of: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Send surviving candidate edges to the LLM for ``kind`` classification.

    The LLM **may not invent edges** — it only labels supplied pairs. Pairs it
    omits keep their deterministic default kind. Pairs it returns ``unknown``
    for get the deterministic default from :func:`typing_default_for`.

    ``cluster_of`` maps paper_id -> cluster label, used to determine whether a
    citation edge crosses cluster boundaries (affects the ``unknown`` default).
    """
    if not edges:
        return edges

    for edge in edges:
        edge["context"] = _edge_context(edge, papers_by_id)

    candidates: list[tuple[str, str, str, str]] = [
        (e["src_paper_id"], e["dst_paper_id"], e["how"], e["context"]) for e in edges
    ]
    valid_list = sorted(valid_ids)
    prompt = build_edge_typing_prompt(topic, candidates, valid_list)

    result: EdgeTyping | None = None
    try:
        result = completer.complete_json(
            system=EDGE_TYPING_SYSTEM_PROMPT,
            user=prompt,
            schema=EdgeTyping,
            stage="edge-typing",
        )
    except Exception as exc:  # noqa: BLE001 — typing must never sink the stage
        logger.warning("Edge typing call failed: %s", exc)
        result = None

    if result is None:
        logger.info(
            "Edge typing unavailable; %d edges keep deterministic defaults", len(edges)
        )
        return _strip_internal(edges)

    typed = parse_typing_result(
        result.model_dump() if hasattr(result, "model_dump") else dict(result)
    )

    typed_kinds: dict[tuple[str, str], str] = {}
    for te in typed:
        typed_kinds[(te.src_paper_id, te.dst_paper_id)] = te.kind

    for edge in edges:
        key = (edge["src_paper_id"], edge["dst_paper_id"])
        kind = typed_kinds.get(key)
        if kind == "unknown" or kind is None:
            cross = False
            if cluster_of is not None:
                s_c = cluster_of.get(edge["src_paper_id"])
                d_c = cluster_of.get(edge["dst_paper_id"])
                cross = s_c is not None and d_c is not None and s_c != d_c
            edge["kind"] = typing_default_for(edge["source"], cross_cluster=cross)
        else:
            edge["kind"] = kind  # type: ignore[assignment]
        edge["rationale"] = ""

    return _strip_internal(edges)


def _strip_internal(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove internal prompt-only fields (``how``, ``context``) before persistence."""
    return [
        {k: v for k, v in edge.items() if k not in ("how", "context")} for edge in edges
    ]


# --------------------------------------------------------------------------- #
# Orchestrator (Task 7.3 — ties the three layers together)
# --------------------------------------------------------------------------- #


def build_graph(
    paper_ids: list[str],
    embeddings: dict[str, np.ndarray],
    citation_triples: list[tuple[str, str, str]],
    topic: str,
    settings: Settings,
    *,
    completer: JSONCompleter | None = None,
    papers_by_id: dict[str, Paper] | None = None,
    cluster_of: dict[str, int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build the full typed edge set for a landscape.

    Returns ``(edges, info)`` where ``info`` records the threshold used and
    per-source counts.

    Pipeline:
        1. kNN layer (cosine >= start threshold)
        2. citation layer (intra-corpus only)
        3. merge (citation beats kNN, undirected dedupe)
        4. density floor (lower cosine threshold if needed)
        5. cap per node
        6. LLM typing (optional — degrades to deterministic defaults)
    """
    from pipeline.embed import embedding_matrix

    matrix = embedding_matrix(paper_ids, embeddings)

    # Layer 2: citations (computed once, reused at every threshold attempt).
    corpus = set(paper_ids)
    cit = citation_edges(citation_triples, corpus)

    # Layers 1+3+4: kNN merged with citations, with density-floor enforcement.
    merged, threshold_used = enforce_density_floor(
        paper_ids, matrix, cit, settings
    )

    # Layer 5: cap per node.
    capped = cap_edges_per_node(merged, settings.graph_edges_per_node)

    # Layer 6: LLM classification.
    if completer is not None and capped:
        typed = classify_edges(
            capped, completer, topic, corpus, papers_by_id or {}, cluster_of or {}
        )
    else:
        typed = _strip_internal(capped)

    info = {
        "threshold_used": threshold_used,
        "density": edge_density(typed, len(paper_ids)),
        "edge_count": len(typed),
        "knn_count": sum(1 for e in typed if e["source"] == "knn"),
        "citation_count": sum(1 for e in typed if e["source"] == "citation"),
    }
    logger.info(
        "Graph built: %d edges (%d kNN, %d citation) at threshold %.2f, density %.2f",
        len(typed),
        info["knn_count"],
        info["citation_count"],
        threshold_used,
        info["density"],
    )
    return typed, info