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
from typing import Any

from config import Settings
from llm.protocol import JSONCompleter
from models import (
    Claims,
    LandscapeSynthesis,
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
    return titles