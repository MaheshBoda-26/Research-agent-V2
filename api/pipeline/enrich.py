"""Stage 2 — citation enrichment (Phase 3, Task 3.4).

Attaches a citation count with provenance to every candidate, and writes
intra-corpus citation links into ``citations`` — only links where **both**
ends are already in the corpus, which is what makes them usable as graph
edges in Phase 7.

Degradation rules (never a per-paper raise):
* S2 first; OpenAlex only as fallback; both failing leaves
  ``citation_count=None`` (unknown — never zero).
* Both sources mocked to fail must return the papers unchanged.

Concurrency: each paper is resolved in ``asyncio.to_thread`` behind a bounded
semaphore of 4. The semaphore bounds *our* concurrency; S2's own 1 RPS pacing
lives in :mod:`sources.semanticscholar`.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any

from config import Settings
from models import Paper
from sources import openalex as openalex_module
from sources import semanticscholar

#: Emitted as ``{"done": int, "total": int}`` after each paper is resolved.
ProgressCallback = Callable[[dict], None]

logger = logging.getLogger(__name__)

_S2_TTL_SECONDS = 7 * 24 * 3600.0


def _s2_paper_id(payload: dict[str, Any]) -> str:
    external = payload.get("externalIds") or {}
    s2id = payload.get("paperId")
    return str(external.get("S2PaperId") or s2id or "")


def _s2_referenced_arxiv_ids(payload: dict[str, Any]) -> list[str]:
    external = payload.get("externalIds") or {}
    arxiv_id = external.get("ArXiv")
    return [str(arxiv_id)] if arxiv_id else []


def _enrich_one(
    paper: Paper,
    settings: Settings,
    *,
    s2: Any,
    openalex_src: Any,
    conn: Any | None,
) -> Paper:
    """Resolve identity + citation count for one paper. Never raises."""
    updated = paper
    try:
        s2_payload = s2.fetch_paper(paper.paper_id, settings, conn=conn)
        if s2_payload and s2_payload.get("citationCount") is not None:
            updated = updated.model_copy(
                update={
                    "citation_count": int(s2_payload["citationCount"]),
                    "citation_source": "semanticscholar",
                    "s2_paper_id": _s2_paper_id(s2_payload),
                }
            )
    except Exception as exc:  # noqa: BLE001 - degrade, never break enrichment
        logger.warning("S2 enrichment failed for %s: %s", paper.paper_id, exc)

    if updated.citation_count is None:
        try:
            work = openalex_src.resolve_work(paper, settings, conn=conn)
            if work and work.get("cited_by_count") is not None:
                updated = updated.model_copy(
                    update={
                        "citation_count": int(work["cited_by_count"]),
                        "citation_source": "openalex",
                        "openalex_id": str(work.get("id") or ""),
                    }
                )
        except Exception as exc:  # noqa: BLE001 - degrade, never break enrichment
            logger.warning("OpenAlex enrichment failed for %s: %s", paper.paper_id, exc)

    return updated


async def enrich_papers(
    papers: list[Paper],
    settings: Settings,
    *,
    s2: Any | None = None,
    openalex: Any | None = None,  # noqa: F811 - shadowing the module name by design (A.6)
    conn: Any | None = None,
    progress: ProgressCallback | None = None,
) -> tuple[list[Paper], list[tuple[str, str, str]]]:
    """Return ``(papers with citation metadata, intra-corpus citation triples)``.

    Triple shape: ``(src_paper_id, dst_paper_id, source)`` where ``src`` cites
    ``dst``. Only links whose both ends are in ``papers`` are returned.
    """
    s2 = s2 if s2 is not None else semanticscholar
    openalex_src = openalex if openalex is not None else openalex_module

    total = len(papers)
    if total == 0:
        return [], []

    semaphore = asyncio.Semaphore(4)
    done = 0

    async def run_one(paper: Paper) -> Paper:
        nonlocal done
        async with semaphore:
            result = await asyncio.to_thread(
                _enrich_one, paper, settings, s2=s2, openalex_src=openalex_src, conn=conn
            )
            done += 1
            if progress is not None:
                progress({"done": done, "total": total})
            return result

    resolved = await asyncio.gather(*(run_one(p) for p in papers))

    # Intra-corpus links: S2 references give arXiv ids directly.
    corpus_ids = {p.paper_id for p in papers}
    triples: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str]] = set()
    for paper in resolved:
        try:
            refs = s2.fetch_references(paper.paper_id, settings, conn=conn)
        except Exception as exc:  # noqa: BLE001 - degrade
            logger.warning("S2 reference fetch failed for %s: %s", paper.paper_id, exc)
            refs = []
        for ref in refs:
            cited = ((ref.get("citedPaper") or ref).get("externalIds") or {}).get("ArXiv")
            if cited and cited in corpus_ids and (paper.paper_id, cited) not in seen:
                seen.add((paper.paper_id, cited))
                triples.append((paper.paper_id, cited, "semanticscholar"))

    return list(resolved), triples
