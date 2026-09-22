"""B.1 — translate a plain-English topic into an arXiv ``search_query``.

The arXiv API is not a natural-language interface: it wants field-prefixed
boolean expressions such as ``all:"retrieval-augmented generation" AND
cat:cs.CL``. An LLM decomposes the topic first (tier 1); if that fails or
returns nothing usable the pipeline degrades to a quoted-phrase search of the
topic verbatim (tier 2), then to the same search unquoted (tier 3).

If no LLM is available the pipeline still runs — every tier is a plain string,
so the whole resolver is testable without a key (plan §11.2).
"""

from __future__ import annotations

import re
from typing import ClassVar

#: Categories that actually contain ML work. Used as a hard constraint on the
#: LLM tier: a hallucinated category silently reduces recall to zero, which
#: reads as a broken app rather than a bad model guess (B.1 rules).
ML_CATEGORIES: tuple[str, ...] = (
    "cs.AI",
    "cs.CL",
    "cs.CV",
    "cs.IR",
    "cs.LG",
    "cs.RO",
    "cs.NE",
    "stat.ML",
)

QUERY_SYSTEM_PROMPT = """You translate a plain-English machine-learning topic into one arXiv search_query string.

Return JSON: {"search_query": str, "rationale": str}

Grammar rules for search_query (nothing else is allowed):
- Field prefixes: all: (any field), ti: (title), abs: (abstract), cat: (category).
- Boolean operators: AND, OR (uppercase).
- Exact phrases in double quotes: all:"retrieval-augmented generation".
- Categories must come from this list only: %s.
- At least 2 terms. Quote the topic's distinctive phrase(s); do not search the
  raw topic as one loose string.
- No sortBy, no pagination, no URLs.

Example for "making AI remember things":
{"search_query": "all:\\"retrieval-augmented generation\\" OR all:\\"long-term memory\\" AND (cat:cs.CL OR cat:cs.LG)", "rationale": "canonical RAG phrasing plus memory systems, restricted to NLP/ML"}

Respond with JSON only.""" % ", ".join(ML_CATEGORIES)


def build_user_prompt(topic: str) -> str:
    return f"Topic: {topic}"


_WHITESPACE = re.compile(r"\s+")
_ALLOWED_PREFIXES = re.compile(r"(?:all|ti|abs|cat):")
_CATEGORY_TERM = re.compile(r"cat:([A-Za-z][A-Za-z0-9.-]*)")
_FORBIDDEN = re.compile(r"sortby|http|start=|max_results", re.IGNORECASE)


def _clean_term(term: str) -> str:
    """Strip characters that would break the arXiv query grammar."""
    return _WHITESPACE.sub(" ", term).strip().strip('"').strip()


def heuristic_query(topic: str) -> str:
    """Tier 2 — no-LLM fallback: the topic as one exact quoted phrase.

    Deliberately conservative: high precision, and if it returns nothing
    ``fetch_candidates`` retries with ``plain_query``.
    """
    cleaned = _clean_term(topic)
    return f'all:"{cleaned}"' if cleaned else ""


def plain_query(topic: str) -> str:
    """Tier 3 — the unquoted retry used when the quoted phrase finds nothing."""
    cleaned = _clean_term(topic)
    return f"all:{cleaned}" if cleaned else ""


def is_wellformed_query(query: str) -> bool:
    """B.1 acceptance rules for an LLM-proposed query string.

    Malformed output must fall through to tier 2, never reach arXiv: a query
    with an invented category or a sortBy parameter silently returns nothing
    (categories) or is rejected outright (syntax).
    """
    cleaned = _clean_term(query)
    if not cleaned or _FORBIDDEN.search(cleaned):
        return False
    if not _ALLOWED_PREFIXES.search(cleaned):
        return False
    # At least two field-prefixed terms (B.1: >=2 terms).
    if len(_ALLOWED_PREFIXES.findall(cleaned)) < 2:
        return False
    # Every category must be one we know exists in the ML literature.
    for category in _CATEGORY_TERM.findall(cleaned):
        if category not in ML_CATEGORIES:
            return False
    return True


class QueryBuilderVersions:
    """Marker so the query prompt participates in prompt versioning."""

    CURRENT: ClassVar[str] = "query_v2"
