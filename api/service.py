"""Assemble API responses from the database.

Kept out of ``main.py`` so the HTTP layer only deals with transport, and so the
response shapes can be built and asserted in tests without spinning up a server.

One derived field is computed here rather than stored: ``relative_score``. It is
a percentile rank within the landscape, so storing it would duplicate information
that is already implied by the persisted logits, and would go stale the moment a
paper were added. It is computed on read from the same function the pipeline
uses, so the CLI and the API agree on what it means.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

import store
from config import Settings
from models import (
    UNCLUSTERED_LABEL,
    ClusterOut,
    EdgeOut,
    LandscapeDetail,
    LandscapeSummary,
    OpenProblemOut,
    PaperInLandscape,
    ReadingStepOut,
    TensionOut,
)
from pipeline.rerank import relative_scores


def _params(row: dict[str, Any]) -> dict[str, Any]:
    raw = row.get("params_json") or "{}"
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _prompt_version(row: dict[str, Any], settings: Settings) -> str:
    return str(_params(row).get("prompt_version") or settings.prompt_version)


def build_summary(row: dict[str, Any]) -> LandscapeSummary:
    return LandscapeSummary(
        id=int(row["id"]),
        topic=row.get("topic") or "",
        title=row.get("title") or "",
        summary=row.get("summary") or "",
        status=row.get("status") or "running",
        narrative_status=row.get("narrative_status") or "pending",
        generation=int(row.get("generation") or 1),
        paper_count=int(row.get("paper_count") or 0),
        cluster_count=int(row.get("cluster_count") or 0),
        cost_usd=float(row.get("cost_usd") or 0.0),
        created_at=row.get("created_at") or "",
        updated_at=row.get("updated_at") or "",
    )


def list_summaries(conn: sqlite3.Connection) -> list[LandscapeSummary]:
    return [build_summary(row) for row in store.list_landscapes(conn)]


def _relative_scores(papers: list[dict[str, Any]]) -> dict[str, float]:
    """Percentile-rank the papers that have logits.

    Papers reranked by the degraded fallback path have no logit, because rank
    order carries no magnitude. They are left without a relative score rather
    than being assigned a misleading one.
    """
    with_logits = [
        (paper["paper_id"], float(paper["cross_encoder_logit"]))
        for paper in papers
        if paper.get("cross_encoder_logit") is not None
    ]
    if not with_logits:
        return {}
    scores = relative_scores([logit for _, logit in with_logits])
    return {paper_id: score for (paper_id, _), score in zip(with_logits, scores, strict=True)}


def build_detail(
    conn: sqlite3.Connection, landscape_id: int, settings: Settings
) -> LandscapeDetail | None:
    row = store.fetch_landscape(conn, landscape_id)
    if row is None:
        return None

    paper_rows = store.fetch_landscape_papers(conn, landscape_id)
    cluster_rows = store.fetch_clusters(conn, landscape_id)
    extractions = store.fetch_extractions(
        conn, [paper["paper_id"] for paper in paper_rows], _prompt_version(row, settings)
    )
    relative = _relative_scores(paper_rows)

    papers = [
        PaperInLandscape(
            paper_id=paper["paper_id"],
            version=paper.get("version") or "",
            title=paper["title"],
            abstract=paper["abstract"],
            authors=paper.get("authors") or [],
            published=paper.get("published") or "",
            updated=paper.get("updated") or "",
            primary_category=paper.get("primary_category") or "",
            categories=paper.get("categories") or [],
            comment=paper.get("comment") or "",
            journal_ref=paper.get("journal_ref") or "",
            doi=paper.get("doi") or "",
            abs_url=paper["abs_url"],
            pdf_url=paper.get("pdf_url") or "",
            citation_count=paper.get("citation_count"),
            citation_source=paper.get("citation_source") or "",
            openalex_id=paper.get("openalex_id") or "",
            s2_paper_id=paper.get("s2_paper_id") or "",
            fulltext_status=paper.get("fulltext_status") or "none",
            rank=int(paper["rank"]),
            relevance_score=float(paper["relevance_score"]),
            cross_encoder_logit=paper.get("cross_encoder_logit"),
            relative_score=relative.get(paper["paper_id"]),
            rerank_source=paper.get("rerank_source") or "cross-encoder",
            cluster_id=paper.get("cluster_id"),
            x=float(paper["x"] or 0.0),
            y=float(paper["y"] or 0.0),
            is_seed=bool(paper.get("is_seed")),
            extraction=extractions.get(paper["paper_id"]),
            extraction_status=(
                extractions[paper["paper_id"]].status
                if paper["paper_id"] in extractions
                else "failed"
            ),
        )
        for paper in paper_rows
    ]

    clusters = [
        ClusterOut(
            id=int(cluster["id"]),
            label=cluster.get("label") or "",
            description=cluster.get("description") or "",
            paper_count=int(cluster.get("paper_count") or 0),
            x=float(cluster.get("x") or 0.0),
            y=float(cluster.get("y") or 0.0),
            color=cluster.get("color") or "#94a3b8",
            is_unclustered=int(cluster.get("local_label", 0)) == UNCLUSTERED_LABEL,
        )
        for cluster in cluster_rows
    ]

    reading_path = [
        ReadingStepOut(
            paper_id=step["paper_id"],
            position=int(step["position"]),
            why=step.get("why") or "",
            title=step.get("title") or "",
        )
        for step in store.fetch_reading_path(conn, landscape_id)
    ]

    return LandscapeDetail(
        id=int(row["id"]),
        topic=row.get("topic") or "",
        title=row.get("title") or "",
        summary=row.get("summary") or "",
        status=row.get("status") or "running",
        narrative_status=row.get("narrative_status") or "pending",
        generation=int(row.get("generation") or 1),
        cost_usd=float(row.get("cost_usd") or 0.0),
        created_at=row.get("created_at") or "",
        updated_at=row.get("updated_at") or "",
        clusters=clusters,
        papers=papers,
        edges=[
            EdgeOut(
                src_paper_id=edge["src_paper_id"],
                dst_paper_id=edge["dst_paper_id"],
                kind=edge["kind"],
                weight=float(edge["weight"]),
                rationale=edge.get("rationale") or "",
                source=edge.get("source") or "llm",
                confidence=edge.get("confidence"),
            )
            for edge in store.fetch_edges(conn, landscape_id)
        ],
        tensions=[
            TensionOut(
                statement=tension["statement"],
                paper_a_id=tension["paper_a_id"] or "",
                paper_b_id=tension["paper_b_id"] or "",
            )
            for tension in store.fetch_tensions(conn, landscape_id)
        ],
        open_problems=[
            OpenProblemOut(
                statement=problem["statement"],
                why_open=problem.get("why_open") or "",
                supporting_paper_ids=problem.get("supporting_paper_ids") or [],
            )
            for problem in store.fetch_open_problems(conn, landscape_id)
        ],
        reading_path=reading_path,
    )


def fetch_paper_detail(
    conn: sqlite3.Connection, landscape_id: int, paper_id: str, settings: Settings
) -> PaperInLandscape | None:
    detail = build_detail(conn, landscape_id, settings)
    if detail is None:
        return None
    for paper in detail.papers:
        if paper.paper_id == paper_id:
            return paper
    return None


__all__ = [
    "build_detail",
    "build_summary",
    "fetch_paper_detail",
    "list_summaries",
]
