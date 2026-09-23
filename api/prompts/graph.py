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

#: System prompt for edge typing (B.5). Rules and the kind enum live here;
#: untrusted paper text lives only in the user message (B.9 rule 1).
EDGE_TYPING_SYSTEM_PROMPT = """You classify relationships between research papers.

For each pair of papers the user supplies, assign exactly one relationship kind.

Allowed kinds (exactly these four strings):
  extends       — dst builds directly on src's method/result
  contradicts   — the two papers' stated claims/results conflict
  applies       — both address the same problem with different methods
  shares_method — both use a similar technique/approach
  unknown       — not enough to decide (maps to a default below)

Rules:
- Use contradicts ONLY when findings/claims actually conflict.
  Different approaches to the same problem is NOT a contradiction.
- Use extends when dst builds on, improves, or generalizes src.
- Use applies when they tackle the same problem differently.
- Use shares_method when the shared technique is the notable link.
- Prefer a confident kind over 'unknown' when the evidence supports it.
- Classify ONLY the pairs the user supplies — never invent a new pair.
- Every src/dst id you return must come from the user's VALID PAPER IDS list.

Return JSON only:
{"edges": [{"src_paper_id": "...", "dst_paper_id": "...", "kind": "..."}, ...]}"""


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
    """Build the user prompt carrying the untrusted data (B.9 rule 1).

    The classification rules and the kind enum live in
    :data:`EDGE_TYPING_SYSTEM_PROMPT`; this message carries only the topic, the
    valid id allow-list, and the candidate pairs with their evidence snippets.

    ``candidates`` is a list of ``(src_id, dst_id, how_we_know, context)`` tuples
    where ``how_we_know`` is 'kNN' or 'citation' and ``context`` is a short
    snippet describing both papers.
    """
    lines = [
        f"TOPIC: {topic}",
        "",
        "VALID PAPER IDS (the only ids you may reference):",
        ", ".join(valid_ids),
        "",
        "CANDIDATE EDGES (classify each):",
        "",
    ]

    for src, dst, how, context in candidates:
        lines.append(f'<pair src="{src}" dst="{dst}">')
        lines.append(f"  known via: {how}")
        if context:
            lines.append(f"  {context}")
        lines.append("</pair>")
        lines.append("")

    return "\n".join(lines)


#: The four kinds the LLM may legitimately assign (B.5). ``unknown`` and any
#: other string are *not* persisted — they are dropped here so the caller maps
#: the candidate to its deterministic default.
_VALID_KINDS = frozenset({"extends", "contradicts", "applies", "shares_method"})


def parse_typing_result(raw: dict) -> list[TypedEdge]:
    """Parse the LLM's edge-typing envelope, per-edge lenient (B.5).

    Each entry must carry non-empty string ``src_paper_id``/``dst_paper_id`` and
    a ``kind``. Entries whose kind is not one of the four valid kinds — including
    the allowed ``"unknown"`` — are dropped; the caller treats a missing entry
    exactly like ``unknown`` and applies the deterministic default. A wholly
    malformed envelope yields an empty list (every candidate defaults).
    """
    if not isinstance(raw, dict):
        return []
    edges = raw.get("edges")
    if not isinstance(edges, list):
        return []
    out: list[TypedEdge] = []
    for item in edges:
        if not isinstance(item, dict):
            continue
        src = item.get("src_paper_id")
        dst = item.get("dst_paper_id")
        kind = item.get("kind")
        if not (isinstance(src, str) and src and isinstance(dst, str) and dst):
            continue
        if kind not in _VALID_KINDS:
            continue  # "unknown" or garbage → caller applies the default
        out.append(TypedEdge(src_paper_id=src, dst_paper_id=dst, kind=kind))
    return out

