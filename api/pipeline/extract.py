"""Stage 4 — structured extraction with a coverage ledger (Phase 5, Task 5.3).

One LLM call per paper turns an abstract (plus optional full-text sections)
into problem, method, results, contribution, limitations, and a novelty
assessment — then a per-paper ledger records whether each paper made it.

The interesting part is the evidence check, ported verbatim from V1's
``pipeline/extract.py``. Every populated prose field must be backed by a quote
from the source text, and that quote is verified as a literal substring before
the field is stored. A field that cannot be grounded is nulled rather than
displayed, because a map that describes papers inaccurately is worse than a
map with holes in it.

Concurrency is ``asyncio.to_thread`` behind a semaphore sized to
``LLM_CONCURRENCY`` (default 4): bounded, never a thread per paper. Each paper
gets a per-paper timeout and ONE retry, then ``status='failed'`` with a
non-empty reason. Both ok and failed rows are persisted via :func:`persist`
keyed by ``(paper_id, prompt_version)`` — this differs from V1, which refused
to cache empty rows; here a failed row is an explicit "attempted and failed"
marker, so a rerun with an unchanged ``PROMPT_VERSION`` issues zero LLM calls.

Progress events are single dicts (the enrich-stage convention): ::

    {"event": "extract_progress", "done": d, "total": t}
"""

from __future__ import annotations

import asyncio
import logging
import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import store
from config import Settings
from llm.protocol import JSONCompleter
from models import Paper, PaperExtraction
from prompts.extract import (
    EXTRACT_SYSTEM_PROMPT,
    build_correction_prompt,
    build_user_prompt,
)

logger = logging.getLogger(__name__)

#: Progress events are single dicts, e.g. ``{"event": ..., "done": 3, "total": 20}``.
ProgressCallback = Callable[[dict], None]

#: The ``event`` value carried by every progress dict from this stage.
EXTRACT_PROGRESS_EVENT = "extract_progress"


@dataclass(frozen=True)
class Coverage:
    """Per-paper ledger totals for one extraction run (plan Appendix A.6)."""

    ok: int
    failed: int
    skipped: int

    @property
    def ratio(self) -> float:
        """Share of papers with a usable extraction (``ok / total``)."""
        total = self.ok + self.failed + self.skipped
        if total == 0:
            return 1.0
        return self.ok / total


#: A quote this short is a substring by accident, not evidence.
MIN_EVIDENCE_CHARS = 16

#: Characters that vary by typography rather than by meaning. Abstracts mix
#: LaTeX, Unicode dashes, and smart quotes, so a verbatim quote frequently fails
#: a naive substring test for reasons that have nothing to do with the model.
_PUNCTUATION_MAP = {
    " ": "-",
    "‑": "-",
    "‒": "-",
    "–": "-",
    "—": "-",
    "−": "-",
    "‘": "'",
    "’": "'",
    "‚": "'",
    "“": '"',
    "”": '"',
    " ": " ",
    " ": " ",
    " ": " ",
}

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_for_match(text: str) -> str:
    """Normalize text for substring comparison, preserving wording exactly."""
    if not text:
        return ""
    folded = unicodedata.normalize("NFKC", text)
    # NOTE: verbatim V1 normalization; the ignore covers dict-invariance noise.
    translated = folded.translate(str.maketrans(_PUNCTUATION_MAP))  # type: ignore[arg-type]
    collapsed = _WHITESPACE_RE.sub(" ", translated)
    return collapsed.strip().casefold()


def is_verbatim(quote: str | None, abstract: str) -> bool:
    """True when ``quote`` appears in ``abstract``, ignoring typography only."""
    if not quote or len(quote.strip()) < MIN_EVIDENCE_CHARS:
        return False
    return normalize_for_match(quote) in normalize_for_match(abstract)


def ungrounded_fields(extraction: PaperExtraction, abstract: str) -> list[str]:
    """Names of populated prose fields whose evidence quote is not verbatim.

    A non-null field with no evidence entry at all is also ungrounded: the
    contract requires a quote, and silence is not a quote.
    """
    bad: list[str] = []
    for field in PaperExtraction.EXTRACTED_FIELDS:
        value = getattr(extraction, field, None)
        if value is None or not str(value).strip():
            continue
        if not is_verbatim(extraction.evidence.get(field), abstract):
            bad.append(field)
    return bad


def enforce_evidence(extraction: PaperExtraction, abstract: str) -> PaperExtraction:
    """Null any field that could not be grounded, and drop stale evidence keys."""
    bad = set(ungrounded_fields(extraction, abstract))
    if not bad:
        return extraction

    updates: dict[str, object] = dict(extraction.model_dump())
    for field in bad:
        updates[field] = None
    updates["evidence"] = {key: value for key, value in extraction.evidence.items() if key not in bad}
    logger.info("Ungrounded fields nulled for a paper: %s", sorted(bad))
    return PaperExtraction(**updates)  # type: ignore[arg-type]


def has_grounded_content(extraction: PaperExtraction) -> bool:
    """True when at least one prose field survived grounding."""
    return any(getattr(extraction, field, None) for field in PaperExtraction.EXTRACTED_FIELDS)


def unextracted(paper_id: str = "") -> PaperExtraction:
    """The stored result of a failed extraction.

    An explicit nulled row rather than a missing one, so a re-run can tell
    "attempted and failed" apart from "never attempted" without another call.
    """
    _ = paper_id
    return PaperExtraction(novelty="unclear", status="failed", error="extraction failed")


def _source_text(paper: Paper, fulltext: dict[str, dict[str, str]] | None) -> str:
    """Text the model was shown: abstract plus any cached full-text sections."""
    sections = (fulltext or {}).get(paper.paper_id) or {}
    extra = "\n\n".join(text for text in sections.values() if text and text.strip())
    if extra:
        return f"{paper.abstract}\n\n{extra}"
    return paper.abstract


def _failure(paper_id: str, reason: str) -> PaperExtraction:
    """A ``status='failed'`` row with a non-empty reason. Never ``ok``."""
    _ = paper_id
    return PaperExtraction(novelty="unclear", status="failed", error=reason or "extraction failed")


def _call_completer(
    completer: JSONCompleter,
    *,
    system: str,
    user: str,
    stage: str = "extraction",
) -> PaperExtraction | None:
    """One blocking model call, tolerating V1-shaped test doubles.

    The :class:`JSONCompleter` protocol carries ``stage``/``run_id``/``attempt``,
    but scripted fakes in the V1 test lineage accept only
    ``(system, user, schema, temperature)``. ``stage`` is passed when the
    completer accepts it.
    """
    try:
        result = cast(
            Any,
            completer.complete_json(system=system, user=user, schema=PaperExtraction, stage=stage),  # type: ignore[call-arg]
        )
    except TypeError:
        result = cast(Any, completer.complete_json(system=system, user=user, schema=PaperExtraction))  # type: ignore[call-arg]
    if result is None:
        return None
    if isinstance(result, PaperExtraction):
        return result
    return PaperExtraction.model_validate(result)


def extract_one(
    paper: Paper,
    completer: JSONCompleter | None,
    settings: Settings,
    *,
    fulltext: dict[str, dict[str, str]] | None = None,
    corrective_retries: int = 1,
) -> PaperExtraction:
    """Extract structure from one paper, grounding every field (sync core)."""
    if completer is None:
        return _failure(paper.paper_id, "no LLM completer available")
    if not paper.abstract or not paper.abstract.strip():
        return _failure(paper.paper_id, "empty abstract: nothing to extract")

    source = _source_text(paper, fulltext)
    user = build_user_prompt(paper)
    try:
        result = _call_completer(completer, system=EXTRACT_SYSTEM_PROMPT, user=user)
    except Exception as exc:  # noqa: BLE001 - a failed paper, not a failed run
        logger.warning("Extraction failed for %s: %s", paper.paper_id, exc)
        return _failure(paper.paper_id, f"completer raised {type(exc).__name__}: {exc}")
    if result is None:
        logger.warning("Extraction failed for %s; storing a null record", paper.paper_id)
        return _failure(paper.paper_id, "completer returned no result")

    bad = ungrounded_fields(result, source)
    attempts = 0
    while bad and attempts < corrective_retries:
        attempts += 1
        logger.info("Re-prompting for %s: %d ungrounded field(s)", paper.paper_id, len(bad))
        try:
            retry = _call_completer(
                completer,
                system=EXTRACT_SYSTEM_PROMPT,
                user=f"{user}\n\n{build_correction_prompt(bad)}",
            )
        except Exception as exc:  # noqa: BLE001 - degrade per paper
            logger.warning("Extraction retry failed for %s: %s", paper.paper_id, exc)
            return _failure(paper.paper_id, f"retry raised {type(exc).__name__}: {exc}")
        if retry is None:
            return _failure(
                paper.paper_id,
                f"retry returned no result after ungrounded fields: {', '.join(sorted(bad))}",
            )
        result = retry
        bad = ungrounded_fields(result, source)

    grounded = enforce_evidence(result, source)
    if bad:
        # Still ungrounded after the correction pass: an explicit failed row
        # with the validation error appended, never a silent partial-ok.
        return _failure(paper.paper_id, f"ungrounded fields after retry: {', '.join(sorted(bad))}")
    if not has_grounded_content(grounded):
        return _failure(paper.paper_id, "no grounded content after evidence enforcement")
    grounded = grounded.model_copy(update={"status": "ok", "error": ""})
    return grounded


def persist(
    extraction: PaperExtraction,
    paper_id: str,
    prompt_version: str,
    settings: Settings,
) -> None:
    """Persist one extraction row (ok or failed) under ``settings.db_path``.

    Failed rows are cached deliberately: a rerun with an unchanged
    ``PROMPT_VERSION`` must issue zero LLM calls. The ``papers`` row is ensured
    first so the ``paper_extractions`` foreign key never fails.
    """
    import sqlite3

    conn = sqlite3.connect(str(settings.db_path), timeout=30.0)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        row = conn.execute("SELECT paper_id FROM papers WHERE paper_id = ?", (paper_id,)).fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO papers (paper_id, title, abstract, fetched_at) VALUES (?, ?, ?, ?)",
                (paper_id, paper_id, "", store.utcnow()),
            )
        store.upsert_extraction(conn, paper_id, prompt_version, extraction, model="")
        conn.commit()
    finally:
        conn.close()


async def persist_async(
    extraction: PaperExtraction,
    paper_id: str,
    prompt_version: str,
    settings: Settings,
) -> None:
    """Thread-offloaded :func:`persist` — never blocks the event loop."""
    await asyncio.to_thread(persist, extraction, paper_id, prompt_version, settings)


async def _extract_one_with_timeout(
    paper: Paper,
    completer: JSONCompleter | None,
    settings: Settings,
    *,
    fulltext: dict[str, dict[str, str]] | None,
) -> PaperExtraction:
    """Run :func:`extract_one` in a worker thread under the per-paper timeout."""
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(extract_one, paper, completer, settings, fulltext=fulltext),
            timeout=settings.extract_timeout_seconds,
        )
    except asyncio.TimeoutError:
        logger.warning("Extraction timed out for %s", paper.paper_id)
        return _failure(paper.paper_id, f"extraction timed out after {settings.extract_timeout_seconds}s")


def extract_one_via_prompt(
    paper: Paper,
    completer: JSONCompleter,
    settings: Settings,
    user: str,
    fulltext: dict[str, dict[str, str]] | None,
) -> PaperExtraction | None:
    """Single retry attempt against an explicit prompt; ``None`` = still bad."""
    _ = settings
    source = _source_text(paper, fulltext)
    try:
        result = _call_completer(completer, system=EXTRACT_SYSTEM_PROMPT, user=user)
    except Exception as exc:  # noqa: BLE001 - per-paper degradation
        return _failure(paper.paper_id, f"retry raised {type(exc).__name__}: {exc}")
    if result is None:
        return None
    bad = ungrounded_fields(result, source)
    if bad:
        return _failure(paper.paper_id, f"ungrounded fields after retry: {', '.join(sorted(bad))}")
    grounded = enforce_evidence(result, source)
    if not has_grounded_content(grounded):
        return _failure(paper.paper_id, "no grounded content after evidence enforcement")
    return grounded.model_copy(update={"status": "ok", "error": ""})


async def _read_cache(
    settings: Settings,
    paper_ids: list[str],
    prompt_version: str,
) -> dict[str, PaperExtraction]:
    """Fetch cached rows for this prompt version without blocking the loop."""

    def _read() -> dict[str, PaperExtraction]:
        conn = store.connect(settings.db_path)
        try:
            return store.fetch_extractions(conn, paper_ids, prompt_version)
        finally:
            conn.close()

    return await asyncio.to_thread(_read)


async def extract_all(
    papers: list[Paper],
    settings: Settings,
    *,
    completer: JSONCompleter | None = None,
    fulltext: dict[str, dict[str, str]] | None = None,
    progress: ProgressCallback | None = None,
) -> tuple[dict[str, PaperExtraction], Coverage]:
    """Extract every paper and ledger every outcome (plan Appendix A.6)."""
    total = len(papers)
    if total == 0:
        return {}, Coverage(ok=0, failed=0, skipped=0)

    prompt_version = settings.prompt_version
    cached = await _read_cache(settings, [p.paper_id for p in papers], prompt_version)
    results: dict[str, PaperExtraction] = dict(cached)
    todo = [p for p in papers if p.paper_id not in results]
    done = len(results)
    if progress is not None and done:
        progress({"event": EXTRACT_PROGRESS_EVENT, "done": done, "total": total})

    if completer is None and todo:
        for paper in todo:
            results[paper.paper_id] = _failure(paper.paper_id, "no LLM completer available")
        if progress is not None:
            progress({"event": EXTRACT_PROGRESS_EVENT, "done": total, "total": total})
        ok = sum(1 for e in results.values() if e.status == "ok")
        failed = sum(1 for e in results.values() if e.status == "failed")
        return results, Coverage(ok=ok, failed=failed, skipped=total - ok - failed)

    semaphore = asyncio.Semaphore(max(1, settings.llm_concurrency))

    async def run_one(paper: Paper) -> tuple[str, PaperExtraction]:
        nonlocal done
        async with semaphore:
            outcome = await _extract_one_with_timeout(paper, completer, settings, fulltext=fulltext)
            if outcome.status == "failed" and completer is not None:
                retry_user = (
                    f"{build_user_prompt(paper)}\n\nValidation error from the previous "
                    f"attempt: {outcome.error}. Re-read the abstract and return the full "
                    "JSON object again with verbatim evidence spans."
                )
                try:
                    retried = await asyncio.wait_for(
                        asyncio.to_thread(extract_one_via_prompt, paper, completer, settings, retry_user, fulltext),
                        timeout=settings.extract_timeout_seconds,
                    )
                    if retried is not None:
                        outcome = retried
                except asyncio.TimeoutError:
                    outcome = _failure(paper.paper_id, f"{outcome.error}; retry timed out")
                except Exception as exc:  # noqa: BLE001 - per-paper degradation
                    outcome = _failure(paper.paper_id, f"{outcome.error}; retry raised {type(exc).__name__}")
            await persist_async(outcome, paper.paper_id, prompt_version, settings)
            done += 1
            if progress is not None:
                progress({"event": EXTRACT_PROGRESS_EVENT, "done": done, "total": total})
            return paper.paper_id, outcome

    for paper in todo:
        paper_id, outcome = await run_one(paper)
        results[paper_id] = outcome

    ok = sum(1 for e in results.values() if e.status == "ok")
    failed = sum(1 for e in results.values() if e.status == "failed")
    skipped = total - ok - failed
    return results, Coverage(ok=ok, failed=failed, skipped=skipped)


__all__ = [
    "EXTRACT_PROGRESS_EVENT",
    "MIN_EVIDENCE_CHARS",
    "Coverage",
    "enforce_evidence",
    "extract_all",
    "extract_one",
    "extract_one_via_prompt",
    "has_grounded_content",
    "is_verbatim",
    "normalize_for_match",
    "persist",
    "persist_async",
    "unextracted",
    "ungrounded_fields",
]
