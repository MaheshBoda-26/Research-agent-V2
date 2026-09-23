"""Stage 6 — synthesis (Phase 7, Tasks 7.4-7.7).

Three bounded LLM calls, each in its own failure domain:

    Call A  -> title + 3-5 sentence summary   (schema: Narrative)
    Call B  -> tensions + open problems       (schema: Claims)
    Call C  -> ordered reading path            (schema: ReadingPath)

Every call degrades independently: a failed call produces a deterministic
fallback, and ``narrative_status`` records which layers used fallbacks.

``narrative_status`` ladder (plan §7.7):
    ok      — title + summary + >=1 of {tensions, open_problems, reading_path}
    partial — some section used a fallback but the core narrative is present
    fallback— the prose (Call A) fell back; title/summary are templated
    failed  — title or summary is missing entirely (never happens — empty
              prose triggers fallback, not failed)
    pending — not yet synthesised
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from typing import Any

from config import Settings
from llm.protocol import JSONCompleter
from models import (
    Claims,
    Narrative,
    NarrativeStatus,
    Paper,
    PaperExtraction,
    ReadingPath,
)
from prompts.synthesize import (
    SYNTHESIS_SYSTEM_PROMPT,
    build_claims_prompt,
    build_narrative_prompt,
    build_reading_path_prompt,
)

logger = logging.getLogger(__name__)

__all__ = [
    "synthesize",
    "fallback_narrative",
    "validate_ids",
    "dense_positions",
    "compute_narrative_status",
    "summarise_exemplar_titles",
    "summarises_title",
]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def fallback_narrative(
    topic: str, clusters: list[dict[str, Any]], paper_count: int, unclustered: int
) -> tuple[str, str]:
    """Deterministic fallback when Call A (prose) fails (B.6, Task 7.4).

    Returns ``(title, summary)``.
    """
    k = len(clusters)
    labels = ", ".join(c.get("label") or f"area {i}" for i, c in enumerate(clusters))
    title = f"Survey of {topic}"[:50]
    summary = (
        f"{paper_count} papers on {topic} group into {k} areas: {labels}. "
        f"{unclustered} remain unclustered."
    )
    return title, summary


def _truncate(text: str, max_chars: int = 120) -> str:
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "\u2026"


def summarise_exemplar_titles(
    papers_by_id: dict[str, Paper], cluster_paper_ids: list[str], limit: int = 5
) -> list[str]:
    """Return up to ``limit`` titles for papers in a cluster (for the prompt)."""
    titles: list[str] = []
    for pid in cluster_paper_ids:
        paper = papers_by_id.get(pid)
        if paper and paper.title.strip():
            titles.append(paper.title.strip())
        if len(titles) >= limit:
            break


def validate_ids(
    items: list[dict[str, Any]],
    id_fields: Sequence[str],
    valid_ids: set[str],
) -> list[dict[str, Any]]:
    """Drop items that reference an id not in ``valid_ids`` (Task 7.5).

    Scalar id fields: an invalid value drops the item. List-valued fields
    (``supporting_paper_ids``) are pruned to valid ids; if a non-empty list
    becomes empty the item is dropped (an emptied tension is removed too).
    """
    kept: list[dict[str, Any]] = []
    for item in items:
        ok = True
        cleaned = dict(item)
        for field in id_fields:
            val = cleaned.get(field)
            if isinstance(val, str):
                if val not in valid_ids:
                    ok = False
                    break
            elif isinstance(val, list):
                pruned = [v for v in val if v in valid_ids]
                if not pruned and val:  # non-empty -> empty: drop
                    ok = False
                    break
                cleaned[field] = pruned
        if ok:
            kept.append(cleaned)
    return kept


def dense_positions(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Re-number ``position`` densely from 1 (Task 7.6).

    A model returning positions 1, 2, 4, 7 must become 1, 2, 3, 4. Steps are
    first ordered by their reported position (stable for ties), then renumbered.
    """
    ordered = sorted(steps, key=lambda s: s.get("position", 0))
    for idx, step in enumerate(ordered, start=1):
        step["position"] = idx
    return ordered


def summarises_title(title: str, topic: str) -> bool:
    """True if ``title`` simply restates the topic (anti-restitution gate D4)."""
    title_tokens = set(re.findall(r"[a-z0-9]+", title.lower()))
    topic_tokens = set(re.findall(r"[a-z0-9]+", topic.lower()))
    if not title_tokens or not topic_tokens:
        return True
    overlap = len(title_tokens & topic_tokens) / len(title_tokens)
    return overlap >= 0.5


# --------------------------------------------------------------------------- #
# Synthesis orchestrator (B.6, B.7, B.8)
# --------------------------------------------------------------------------- #


def synthesize(
    clusters: list[dict[str, Any]],
    papers: list[Paper],
    extractions: dict[str, PaperExtraction | None],
    edges: list[dict[str, Any]],
    topic: str,
    settings: Settings,
    *,
    completer: JSONCompleter | None = None,
) -> dict[str, Any]:
    """Run the three-call synthesis, degrading each call independently.

    Returns a dict with ``title``, ``summary``, ``narrative_status``,
    ``tensions``, ``open_problems``, and ``reading_path`` (plan A.10).
    """
    valid_ids = {p.paper_id for p in papers}
    papers_by_id = {p.paper_id: p for p in papers}

    # Build cluster exemplar titles (for Call A).
    exemplar_titles: dict[int, list[str]] = {}
    for cluster in clusters:
        label = int(cluster.get("local_label", 0))
        ids = cluster.get("paper_ids", [])
        exemplar_titles[label] = summarise_exemplar_titles(papers_by_id, ids)

    # --- Call A: narrative prose (title + summary) ---
    prose_fallback = False
    title, summary = "", ""
    if completer is not None:
        user_a = build_narrative_prompt(topic, clusters, exemplar_titles)
        try:
            response = completer.complete_json(
                system=SYNTHESIS_SYSTEM_PROMPT,
                user=user_a,
                schema=Narrative,
                stage="synthesis-narrative",
            )
        except Exception:  # noqa: BLE001
            response = None
        if (
            response is not None
            and response.title.strip()
            and response.summary.strip()
            and not summarises_title(response.title.strip(), topic)
        ):
            title, summary = response.title.strip(), response.summary.strip()
        else:
            prose_fallback = True
    else:
        prose_fallback = True

    if prose_fallback or not title or not summary:
        title, summary = fallback_narrative(
            topic,
            clusters,
            len(papers),
            sum(1 for c in clusters if int(c.get("local_label", 0)) == -1),
        )
    return titles