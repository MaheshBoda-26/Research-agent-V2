"""Pipeline stage functions.

Phase 6 wires the **layout** stage: embed -> project -> cluster -> name ->
persist. The async ``PipelineRun`` orchestrator that drives every stage
(plan Appendix A.10) arrives in Phase 8 (Task 8.3) and will wrap these
synchronous functions in ``asyncio.to_thread``.

Two load-bearing persistence decisions (Task 6.5, plan 3.3 / 2.9):

* ``clusters.local_label`` is HDBSCAN's label (-1, 0, 1, ...) and is stored
  **separately from the table's row id**. ``landscape_papers.cluster_id``
  always references the row id — never the local label — so re-running the
  stage (which deletes and re-inserts cluster rows) cannot silently rewire
  membership. ``cluster_id_by_label`` is the only sanctioned bridge.
* ``-1`` is persisted as its own row named ``Unclustered`` with the muted
  palette colour. It is a first-class bucket on the map, never dropped and
  never named as a research area (anti-pattern guard, plan Phase 6).

Connections are owned by the caller: every function takes an explicit
``conn`` and never commits (except ``embed_papers``, which caches vectors as
it goes — that cache is idempotent), so the session boundary remains the
single commit point.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Sequence
from typing import Any

from config import Settings
from llm.protocol import JSONCompleter
from models import UNCLUSTERED_LABEL, UNCLUSTERED_NAME, RankedPaper
from pipeline.cluster import (
    ReducerLike,
    cluster_centroids,
    color_for_label,
    group_labels,
    layout,
    name_clusters,
    normalize_coords,
)
from pipeline.embed import EmbedderLike, embed_papers, embedding_matrix
from store import cluster_id_by_label, landscape_paper_ids, replace_clusters, update_layout

logger = logging.getLogger(__name__)


def layout_stage(
    landscape_id: int,
    ranked: Sequence[RankedPaper],
    topic: str,
    settings: Settings,
    conn: sqlite3.Connection,
    *,
    embedder: EmbedderLike | None = None,
    reducer: ReducerLike | None = None,
    clusterer: Any | None = None,
    completer: JSONCompleter | None = None,
) -> dict[str, Any]:
    """Embed, project, cluster, name, and persist one landscape's layout.

    The papers must already be linked into ``landscape_papers`` (the retrieval
    stage owns membership); this function only writes the two layout columns:
    cluster row id, plus the normalised x/y position of every paper.

    Returns the stage payload: ``clusters`` rows exactly as persisted, the
    ``cluster_count`` / ``unclustered_count`` split, and the ``paper_count``
    the SQL invariant is checked against. Naming runs through
    :func:`pipeline.cluster.name_clusters`, so even with no LLM configured
    every real cluster gets a deterministic tf-idf name rather than an empty
    cell; ``-1`` is persisted as ``Unclustered`` and never passed to the
    labeller.
    """
    papers = [item.paper for item in ranked]
    total = len(papers)
    if total == 0:
        logger.warning("Landscape %d: no linked papers; layout skipped", landscape_id)
        return {
            "clusters": [],
            "cluster_count": 0,
            "unclustered_count": 0,
            "paper_count": 0,
        }

    linked = landscape_paper_ids(conn, landscape_id)
    missing = [paper.paper_id for paper in papers if paper.paper_id not in linked]
    if missing:
        raise ValueError(
            f"{len(missing)} paper/papers are not linked to landscape {landscape_id} "
            f"before layout: {missing[:5]} — run the retrieval stage first"
        )

    vectors = embed_papers(papers, settings, conn, embedder=embedder)
    matrix = embedding_matrix([paper.paper_id for paper in papers], vectors)
    result = layout(matrix, settings, reducer=reducer, clusterer=clusterer)
    coords = normalize_coords(result.coords)
    labels = result.labels

    groups = group_labels(labels)
    centroids = cluster_centroids(coords, labels)
    named = {
        entry["local_label"]: entry
        for entry in name_clusters(
            [
                {
                    "local_label": int(label),
                    "size": len(indices),
                    "topic": topic,
                    "paper_ids": [papers[i].paper_id for i in indices],
                }
                for label, indices in sorted(groups.items())
            ],
            papers,
            settings,
            completer=completer,
        )
    }

    rows: list[dict[str, Any]] = []
    for label, indices in sorted(groups.items()):
        if label == UNCLUSTERED_LABEL:
            label_text, description = UNCLUSTERED_NAME, "Below the clustering threshold."
        else:
            entry = named.get(int(label), {})
            label_text = str(entry.get("label") or f"Cluster {label}")
            description = str(entry.get("description") or "")
        rows.append(
            {
                "local_label": int(label),
                "label": label_text,
                "description": description,
                "paper_count": len(indices),
                "x": float(centroids[int(label)][0]),
                "y": float(centroids[int(label)][1]),
                "color": color_for_label(int(label)),
            }
        )

    # Row ids are re-issued on every run (replace = DELETE + INSERT), so the
    # bridge from local label to row id must be read *after* the replace.
    replace_clusters(conn, landscape_id, rows)
    row_id_by_label = cluster_id_by_label(conn, landscape_id)
    update_layout(
        conn,
        landscape_id,
        {
            papers[index].paper_id: (
                float(coords[index][0]),
                float(coords[index][1]),
                row_id_by_label[int(labels[index])],
            )
            for index in range(total)
        },
    )

    cluster_count = sum(1 for row in rows if row["local_label"] != UNCLUSTERED_LABEL)
    unclustered_count = int(sum(row["paper_count"] for row in rows if row["local_label"] == UNCLUSTERED_LABEL))
    logger.info(
        "Landscape %d laid out: %d papers, %d cluster(s), %d unclustered",
        landscape_id,
        total,
        cluster_count,
        unclustered_count,
    )
    return {
        "clusters": rows,
        "cluster_count": cluster_count,
        "unclustered_count": unclustered_count,
        "paper_count": total,
    }


__all__ = ["layout_stage"]
