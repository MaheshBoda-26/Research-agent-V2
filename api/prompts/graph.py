"""Edge typing prompt (B.5).

The LLM classifies candidate edges that were already built by deterministic layers
(kNN cosine, citation links). It may not invent new edges — it only assigns a kind
to pairs we already believe are related.

Each candidate is presented as (src_paper_id, dst_paper_id, how_we_know_this, context).
The context is a short evidence snippet so the model can decide between the four
allowed kinds without hallucinating content.

UNKNOWN handling: if the model returns kind="unknown" for a candidate, the caller
maps it to a deterministic default — "shares_method" for kNN-derived candidates
and "applies" for citation-derived candidates that cross cluster boundaries. The
mapping lives in pipeline.graph.typing_default_for.
"""

from __future__ import annotations

from models import Paper, TypedEdge

#: Maximum characters of a paper's title + abstract preview shown per candidate edge.
EDGE_CONTEXT_CHARS = 300


def _paper_context(paper: Paper) -> str:
    """Short context snippet for one paper (title + truncated abstract)."""
    title = paper.title.strip()
    abstract = " ".join((paper.abstract or "").split())
    if len(abstract) > EDGE_CONTEXT_CHARS:
        abstract = abstract[: EDGE_CONTEXT_CHARS - 1] + "…"
    return f"{title}. {abstract}".strip()


def build_edge_typing_prompt(
    topic: str,
    candidates: list[tuple[str, str, str, str]],
    valid_ids: list[str],
) -> str:
    """Build the LLM prompt that classifies candidate edges.

    candidates is a list of (src_id, dst_id, how_we_know, context) tuples
    where how_we_know is 'kNN' or 'citation' and context is a short snippet
    describing both papers.
    """
    lines = [
        "You classify relationships between research papers.",
        "",
        f"TOPIC: {topic}",
        "",
        "For each pair of papers below, assign exactly one relationship kind.",
        "",
        "Allowed kinds (exactly these four strings):",
        "  extends       — dst builds directly on src's method/result",
        "  contradicts   — the two papers' stated claims/results conflict",
        "  applies       — both address the same problem with different methods",
        "  shares_method — both use a similar technique/approach",
        "  unknown       — not enough to decide (maps to a default below)",
        "",
        "Rules:",
        "- Use contradicts ONLY when findings/claims actually conflict.",
        "  Different approaches to the same problem is NOT a contradiction.",
        "- Use extends when dst builds on, improves, or generalizes src.",
        "- Use applies when they tackle the same problem differently.",
        "- Use shares_method when the shared technique is the notable link.",
        "- Prefer a confident kind over 'unknown' when the evidence supports it.",
        "- Every src/dst id must be from the VALID PAPER IDS list below.",
        "",
        "VALID PAPER IDS (the only ids you may reference):",
        ", ".join(valid_ids),
        "",
        "CANDIDATE EDGES (classify each):",
        "",
    ]

    for src, dst, how, context in candidates:
        lines.append(f"PAIR: {src} -> {dst}  (known via: {how})")
        lines.append(f"  {context}")
        lines.append("")

    lines.append(
        "Return JSON only:\n{\"edges\": ["
        "{\"src_paper_id\": \"...\", \"dst_paper_id\": \"...\", "
        "\"kind\": \"...\"}, ...]}"
    )
    return "\n".join(lines)


def parse_typing_result(raw: dict) -> list[TypedEdge]:
    """Parse the LLM's edge-typing envelope, dropping any non-TypedEdge entries."""
    from models import EdgeTyping

    try:
        envelope = EdgeTyping.model_validate(raw)
    except Exception:
        return []
    return list(envelope.edges)

