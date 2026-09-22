"""OpenAlex — keyless citation enrichment with verified identity matching.

Plan §2.3's verified trap governs the design: ``GET /works/doi:10.48550/...``
either 404s or returns a **completely different paper**, so arXiv ids are
never mapped through DataCite DOIs. Instead the work is resolved by exact
title search, and the match is accepted only when the normalised-title
similarity is >= 0.9 *and* the publication year is within +/-1. Anything
looser would silently attach someone else's citation graph to our paper.

Like ``semanticscholar``, every function is best-effort: any failure returns
``None`` or ``[]`` and never raises. ``httpx.MockTransport`` is the test seam.
"""

from __future__ import annotations

import difflib
import logging
import re
from typing import Any

import httpx

from config import Settings
from models import Paper
from store import get_cached, put_cached

logger = logging.getLogger(__name__)

_BASE = "https://api.openalex.org"

#: Title-similarity floor for accepting an identity match (§2.3).
_SIMILARITY_FLOOR: float = 0.9

#: Allowed year drift for accepting an identity match.
_YEAR_SLACK: int = 1

#: Responses are cached for 7 days.
_CACHE_TTL_SECONDS: int = 7 * 24 * 3600

_SELECT_FIELDS = "id,title,doi,cited_by_count,publication_year,ids,referenced_works"

_WHITESPACE = re.compile(r"\s+")
_NON_ALNUM = re.compile(r"[^a-z0-9 ]")


def _normalise_title(title: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace — for similarity."""
    cleaned = _NON_ALNUM.sub("", _WHITESPACE.sub(" ", (title or "").lower()).strip())
    return cleaned


def title_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, _normalise_title(a), _normalise_title(b)).ratio()


def _get(
    settings: Settings,
    path: str,
    *,
    params: dict[str, str] | None = None,
    client: httpx.Client | None = None,
) -> httpx.Response | None:
    """One polite-pool GET; ``None`` on any transport failure (never raises)."""
    mailto = {"mailto": settings.openalex_mailto} if settings.openalex_mailto else {}
    owned = client is None
    if owned:
        client = httpx.Client(base_url=_BASE, timeout=15.0)
    try:
        assert client is not None
        return client.get(path, params={**mailto, **(params or {})})
    except httpx.HTTPError as exc:
        logger.warning("OpenAlex request failed: %s", exc)
        return None
    finally:
        if owned:
            client.close()  # type: ignore[union-attr]


def resolve_work(
    paper: Paper,
    settings: Settings,
    *,
    client: httpx.Client | None = None,
    conn: Any | None = None,
) -> dict[str, Any] | None:
    """Resolve an arXiv paper to an OpenAlex work by verified title match.

    Accepts only a candidate whose normalised title is >= 0.9 similar *and*
    whose publication year is within +/-1. Never constructs a DOI lookup —
    that path is the verified §2.3 trap. Returns ``None`` instead of raising
    on transport failure, non-200, or identity mismatch.
    """
    if not paper.title:
        return None
    key = f"openalex:resolve:{paper.paper_id}"
    if conn is not None:
        cached = get_cached(conn, key, "openalex")
        if cached is not None:
            return cached or None

    response = _get(
        settings,
        "/works",
        params={"search": paper.title, "per-page": "1", "select": _SELECT_FIELDS},
        client=client,
    )
    if response is None or response.status_code != 200:
        return None
    results = response.json().get("results") or []
    if not results:
        return None

    candidate = results[0]
    similarity = title_similarity(paper.title, candidate.get("title") or "")
    year = candidate.get("publication_year")
    paper_year = int(paper.published[:4]) if paper.published[:4].isdigit() else None
    year_ok = (
        year is not None
        and paper_year is not None
        and abs(int(year) - paper_year) <= _YEAR_SLACK
    )
    if similarity < _SIMILARITY_FLOOR or not year_ok:
        logger.info(
            "OpenAlex identity mismatch for %s (similarity=%.2f, year_ok=%s)",
            paper.paper_id, similarity, year_ok,
        )
        if conn is not None:
            put_cached(conn, key, "openalex", {}, _CACHE_TTL_SECONDS)  # negative cache
        return None

    if conn is not None:
        put_cached(conn, key, "openalex", candidate, _CACHE_TTL_SECONDS)
    return candidate


def fetch_citations_by_work_id(
    work_id: str,
    settings: Settings,
    *,
    per_page: int = 200,
    client: httpx.Client | None = None,
    conn: Any | None = None,
) -> list[str]:
    """OpenAlex ids of works citing ``work_id``, or ``[]`` on failure."""
    bare = work_id.rsplit("/", 1)[-1]
    key = f"openalex:cites:{bare}"
    if conn is not None:
        cached = get_cached(conn, key, "openalex")
        if cached is not None and isinstance(cached.get("ids"), list):
            return cached["ids"]  # type: ignore[no-any-return]

    response = _get(
        settings,
        "/works",
        params={"filter": f"cites:{bare}", "per-page": str(per_page), "select": "id"},
        client=client,
    )
    if response is None or response.status_code != 200:
        return []
    ids = [result.get("id", "") for result in response.json().get("results") or []]
    if conn is not None:
        put_cached(conn, key, "openalex", {"ids": ids}, _CACHE_TTL_SECONDS)
    return ids


__all__ = ["fetch_citations_by_work_id", "resolve_work", "title_similarity"]
