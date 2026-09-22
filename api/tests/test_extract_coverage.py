"""Tests for the extraction coverage ledger (Phase 5, Task 5.3).

Offline and deterministic: a fake completer fails for exactly 3 of 20 papers.
The ledger must read 17 ok / 3 failed / ratio 0.85, every failed row must carry
a non-empty ``error``, and no failure path may write ``status='ok'``.
"""

from __future__ import annotations

from dataclasses import replace

from conftest import make_paper

from models import Paper, PaperExtraction
from pipeline.extract import Coverage, extract_all


def _abstract(pid: str) -> str:
    return (
        f"Problem statement for paper {pid} studied here. "
        f"Method construction for paper {pid} built here. "
        f"Results measurement for paper {pid} reported here with 48.7 percent gain. "
        f"Contribution claim for paper {pid} shown here. "
        "The approach is not evaluated on open-ended generation tasks at all."
    )


def _grounded(pid: str) -> PaperExtraction:
    span = f"Results measurement for paper {pid} reported here with 48.7 percent gain."
    assert span in _abstract(pid)
    return PaperExtraction(
        problem=f"Problem statement for paper {pid} studied here.",
        method=f"Method construction for paper {pid} built here.",
        results="Results measurement reported with 48.7 percent gain.",
        contribution=f"Contribution claim for paper {pid} shown here.",
        limitations="The approach is not evaluated on open-ended generation tasks at all.",
        evidence={
            "problem": f"Problem statement for paper {pid} studied here.",
            "method": f"Method construction for paper {pid} built here.",
            "results": span,
            "contribution": f"Contribution claim for paper {pid} shown here.",
            "limitations": "not evaluated on open-ended generation tasks at all.",
        },
    )


class FailThreeCompleter:
    """Grounded answers except for three doomed ids (always ``None``)."""

    def __init__(self, doomed: set[str], calls: list[str]) -> None:
        self.doomed = set(doomed)
        self.calls = calls

    def complete_json(self, *, system, user, schema, **kwargs):  # noqa: ANN001, ANN002, ANN003
        pid = ""
        for line in user.splitlines():
            if line.startswith("TITLE: Title for "):
                pid = line.split("TITLE: Title for ")[1].strip()
        self.calls.append(pid or user[:32])
        if pid in self.doomed:
            return None
        return _grounded(pid)

    def complete_text(self, *, system: str, user: str) -> str | None:  # noqa: ANN001, ANN002, ANN003
        return None


def _papers(n: int = 20) -> list[Paper]:
    out = []
    for i in range(n):
        pid = f"2406.{10000 + i}"
        out.append(make_paper(paper_id=pid, title=f"Title for {pid}", abstract=_abstract(pid)))
    return out


async def test_coverage_ledger_17_ok_3_failed(settings) -> None:
    papers = _papers(20)
    doomed = {p.paper_id for p in papers[:3]}
    calls: list[str] = []
    completer = FailThreeCompleter(doomed, calls)

    results, coverage = await extract_all(papers, settings, completer=completer)

    assert isinstance(coverage, Coverage)
    assert coverage.ok == 17
    assert coverage.failed == 3
    assert coverage.skipped == 0
    assert coverage.ratio == 17 / 20
    assert abs(coverage.ratio - 0.85) < 1e-9
    assert len(results) == 20
    for pid in doomed:
        assert results[pid].status == "failed"
        assert results[pid].error.strip() != ""
    for pid, extraction in results.items():
        if pid not in doomed:
            assert extraction.status == "ok"
            assert extraction.error == ""


async def test_failed_rows_persisted_and_no_failed_as_ok(settings) -> None:
    import store

    papers = _papers(20)
    doomed = {p.paper_id for p in papers[:3]}
    completer = FailThreeCompleter(doomed, [])

    results, _ = await extract_all(papers, settings, completer=completer)

    conn = store.connect(settings.db_path)
    try:
        cached = store.fetch_extractions(conn, [p.paper_id for p in papers], settings.prompt_version)
    finally:
        conn.close()
    assert len(cached) == 20
    for extraction in results.values():
        assert extraction.status in ("ok", "failed")
        if extraction.status == "failed":
            assert extraction.error.strip() != ""
    for extraction in cached.values():
        assert extraction.status in ("ok", "failed")
        if extraction.status == "failed":
            assert extraction.error.strip() != ""
        else:
            assert extraction.error == ""
    for pid in doomed:
        assert cached[pid].status == "failed"


async def test_progress_events_are_single_dicts(settings) -> None:
    papers = _papers(4)
    events: list[dict] = []
    completer = FailThreeCompleter(set(), [])

    await extract_all(papers, settings, completer=completer, progress=events.append)

    assert events
    assert events[-1] == {"event": "extract_progress", "done": 4, "total": 4}
    for event in events:
        assert set(event) == {"event", "done", "total"}
        assert event["event"] == "extract_progress"


async def test_rerun_uses_cache_without_llm_calls(settings) -> None:
    papers = _papers(5)
    first_calls: list[str] = []
    await extract_all(papers, settings, completer=FailThreeCompleter(set(), first_calls))
    assert len(first_calls) > 0

    second_calls: list[str] = []
    results, coverage = await extract_all(papers, settings, completer=FailThreeCompleter(set(), second_calls))

    assert second_calls == []
    assert coverage.ok == 5
    assert coverage.failed == 0
    assert coverage.skipped == 0
    assert all(e.status == "ok" for e in results.values())


async def test_single_retry_appends_validation_error(settings) -> None:
    papers = _papers(1)
    seen: list[str] = []

    class FlakyThenGrounded:
        def complete_json(self, *, system, user, schema, **kwargs):  # noqa: ANN001, ANN002, ANN003
            seen.append(user)
            if len(seen) == 1:
                return PaperExtraction(problem="Invented claim with no quote.", evidence={})
            return _grounded(papers[0].paper_id)

        def complete_text(self, *, system: str, user: str) -> str | None:  # noqa: ANN001
            return None

    test_settings = replace(settings, llm_concurrency=1)
    results, coverage = await extract_all(papers, test_settings, completer=FlakyThenGrounded())

    assert coverage.ok == 1
    assert results[papers[0].paper_id].status == "ok"
    assert len(seen) >= 2
    assert any("not a verbatim substring" in prompt or "Validation error" in prompt for prompt in seen[1:])


def test_coverage_ratio_empty_is_one() -> None:
    assert Coverage(ok=0, failed=0, skipped=0).ratio == 1.0
