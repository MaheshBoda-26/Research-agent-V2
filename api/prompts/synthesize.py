"""Synthesis prompts — Phase 7, Tasks 7.4-7.6 (three LLM calls).

The synthesis stage makes **three** separate calls, each bounded so a single
bad response can never collapse the whole narrative:

* Call A — ``synthesis_narrative``: the prose (title + 3-5 sentence summary).
* Call B — ``synthesis_claims``: tensions and open problems.
* Call C — ``synthesis_reading_path``: ordered reading order.

Each ``build_*_prompt`` returns the *user* message. The system prompt that
accompanies it lives in the same module so the contract is self-contained.
"""

from __future__ import annotations

from typing import Any

from models import Paper, PaperExtraction

#: The system message for all three synthesis calls.
SYNTHESIS_SYSTEM_PROMPT = (
    "You are a research synthesiser. You describe what a body of papers says,\n"
    "where it disagrees, what it leaves undone, and how someone should read it\n"
    "to learn the field. Return JSON matching the requested schema exactly --\n"
    "no prose, no code fences, no markdown."
)

#: Maximum tokens of the abstract snippet shown per paper (plan B.7 budget).
EXTRACTION_SNIPPET_CHARS = 400

#: Maximum characters for a one-line contribution in the reading-path prompt.
CONTRIBUTION_SNIPPET_CHARS = 200


# --------------------------------------------------------------------------- #
# Call A - narrative prose (B.6)
# --------------------------------------------------------------------------- #


def build_narrative_prompt(
    topic: str,
    clusters: list[dict[str, Any]],
    exemplar_titles: dict[int, list[str]],
) -> str:
    """Build the user prompt for the narrative (title + summary) call.

    **In:** topic, cluster labels + descriptions + sizes, and up to 5 exemplar
    titles per cluster.  **Budget:** <= 2.5k in.

    ``clusters`` elements look like::

        {"local_label": 0, "label": "...", "description": "...", "paper_count": 8}

    ``exemplar_titles`` maps ``local_label -> [title, ...]`` (already truncated
    to 5).
    """
    lines = [
        f"TOPIC: {topic}",
        "",
        "You are describing a research landscape. The papers have been grouped",
        "into clusters. For each cluster you have a computed label, a one-sentence",
        "description, the number of papers, and up to five exemplar titles.",
        "",
        "Write a landscape narrative in JSON:",
        '  {"title": str, "summary": str}',
        "",
        "Rules:",
        "  - title: 4-10 words, MUST NOT just restate the topic",
        "  - summary: 3-5 sentences that (1) name the computed areas,",
        "    (2) say what the field has settled, and (3) where it is still moving",
        "  - NO citation of papers by id (prose only)",
        "",
        "Clusters:",
    ]

    for cluster in clusters:
        label = cluster.get("local_label", 0)
        name = cluster.get("label") or ""
        desc = cluster.get("description") or ""
        count = cluster.get("paper_count", 0)
        titles = exemplar_titles.get(label, [])[:5]
        lines.append(f'  Cluster {label}: "{name}" ({count} papers)')
        lines.append(f"    description: {desc}")
        if titles:
            lines.append("    exemplar titles:")
            for t in titles:
                clean = " ".join(t.split())[:80]
                lines.append(f"      - {clean}")
        else:
            lines.append("    exemplar titles: (none)")

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Call B - tensions & open problems (B.7)
# --------------------------------------------------------------------------- #


def _extraction_snippet(extraction: PaperExtraction | None) -> str:
    """Build a ~120-token evidence snippet from one paper's extraction."""
    if extraction is None:
        return ""
    parts = []
    for field in ("problem", "method", "limitations"):
        val = getattr(extraction, field, None)
        if val:
            parts.append(val)
    text = " ".join(parts)
    text = " ".join(text.split())  # collapse whitespace
    if len(text) > EXTRACTION_SNIPPET_CHARS:
        text = text[: EXTRACTION_SNIPPET_CHARS - 1] + "\u2026"
    return text


def build_claims_prompt(
    topic: str,
    papers: list[Paper],
    extractions: dict[str, PaperExtraction | None],
) -> str:
    """Build the user prompt for the claims (tensions + open problems) call.

    **In:** topic + up to 40 papers' extractions (problem/method/limitations,
    truncated to ~120 tokens each).  **Budget:** <= 6k in.
    """
    sampled = papers[:40]
    lines = [
        f"TOPIC: {topic}",
        "",
        "For each paper below you have its title and a truncated extraction of",
        "the problem it addresses, the method it uses, and its stated",
        "limitations. Identify the genuine disagreements (tensions) and the",
        "genuine gaps (open problems) in this body of work.",
        "",
        "Output JSON:",
        '  {"tensions": [{"statement": str, "paper_a_id": str, "paper_b_id": str}],',
        '   "open_problems": [{"statement": str, "why_open": str, "supporting_paper_ids": [str]}]}',
        "",
        "Rules:",
        "  - A tension names TWO supplied papers whose stated findings CONFLICT,",
        "    not merely differ (\"uses method X while Paper B uses method Y\") is NOT a tension",
        "  - An open problem must follow from what these papers LEFT UNDONE,",
        "    not from general knowledge of the field",
        "  - Every id referenced must appear in the list below",
        "  - An empty list is a valid answer (\"no direct conflicts found\")",
        "  - NO prose outside the JSON object",
        "",
        "Papers:",
    ]

    for paper in sampled:
        ext = extractions.get(paper.paper_id)
        snippet = _extraction_snippet(ext)
        lines.append(f"  [{paper.paper_id}] {paper.title.strip()}")
        if snippet:
            lines.append(f"    {snippet}")

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Call C - reading path (B.8)
# --------------------------------------------------------------------------- #


def _contribution_snippet(paper: Paper, extraction: PaperExtraction | None) -> str:
    """One-line contribution from title + extraction contribution field."""
    title = " ".join(paper.title.split())[:CONTRIBUTION_SNIPPET_CHARS]
    contrib = ""
    if extraction is not None:
        contrib = " ".join((extraction.contribution or "").split())[:CONTRIBUTION_SNIPPET_CHARS]
    if contrib:
        return f"{title} - {contrib}"
    return title


def build_reading_path_prompt(
    topic: str,
    papers: list[Paper],
    extractions: dict[str, PaperExtraction | None],
) -> str:
    """Build the user prompt for the reading-path call.

    **In:** topic + 30 papers (id, title, year, citation count, one-line
    contribution).  **Budget:** <= 4k in.
    """
    sampled = papers[:30]
    lines = [
        f"TOPIC: {topic}",
        "",
        "Below are 30 papers from a research landscape, each with an id, title,",
        "year, citation count, and a one-line contribution. Recommend a reading",
        "order that takes a newcomer from foundational/survey work through to",
        "specialised and recent results.",
        "",
        "Output JSON:",
        '  {"steps":[{"paper_id": str, "position": int, "why": str}]}',
        "",
        "Rules:",
        "  - 5-10 steps, starting with accessible foundational/survey work",
        "  - then move to specialised and recent",
        "  - 'why' is ONE clause tied to what that paper gives the reader",
        "  - Every referenced paper_id must appear in the list below",
        "  - NO prose outside the JSON object",
        "",
        "Papers (id, title, year, citations, contribution):",
    ]

    for paper in sampled:
        ext = extractions.get(paper.paper_id)
        contrib = _contribution_snippet(paper, ext)
        year = (paper.published or "")[:4] if paper.published else "?"
        cites = paper.citation_count if paper.citation_count is not None else "?"
        lines.append(
            f'  [{paper.paper_id}] "{contrib}" | year={year} | citations={cites}'
        )

    return "\n".join(lines)


__all__ = [
    "SYNTHESIS_SYSTEM_PROMPT",
    "EXTRACTION_SNIPPET_CHARS",
    "CONTRIBUTION_SNIPPET_CHARS",
    "build_narrative_prompt",
    "build_claims_prompt",
    "build_reading_path_prompt",
]