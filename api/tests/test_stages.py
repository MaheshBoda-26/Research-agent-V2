"""Tests for the V2 orchestrator — plan Phase 8, Task 8.3.

The leaves (arXiv fetch, citation enrichment, cross-encoder, embedder,
clusterer, LLM completer) are injected seams; only ``PipelineRun`` wiring,
the §3.3 failure table, and ``degraded`` propagation are asserted here.

Two plan-mandated cases:

* enrichment raises → the run still reaches ``synthesis`` with
  ``degraded=True``;
* layout raises → the run ends with ``status='failed'`` and a retryable
  payload.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

import store
from config import Settings
from models import Paper, PaperExtraction, RankedPaper
from pipeline import stages as stages_mod
from pipeline.stages import PipelineRun


def _papers(n: int = 6) -> list[Paper]:
    from tests.conftest import make_paper

    return [
        make_paper(
            paper_id=f"2406.{1000 + i}",
            title=f"Paper {i} on test topic",
            abstract=(
                "We study test topic with a controlled experiment. "
                f"Result {i} improves accuracy. Limitation {i} remains."
            ),
        )
        for i in range(n)
    ]


def _ranked(papers: list[Paper]) -> list[RankedPaper]:
    return [
        RankedPaper(
            paper=paper,
            rank=i + 1,
            relevance_score=9.0 - i * 0.1,
            cross_encoder_logit=5.0 - i * 0.1,
            rerank_source="cross-encoder",
        )
        for i, paper in enumerate(papers)
    ]


class _FakeEmbedder:
    model_name = "test-embed"

    def encode(self, texts: list[str]) -> np.ndarray:
        rng = np.random.default_rng(0)
        return rng.normal(size=(len(texts), 8)).astype(np.float32)


class _FakeClusterer:
    def fit_predict(self, _matrix: np.ndarray) -> np.ndarray:
        n = _matrix.shape[0]
        return np.array([0] * (n - 1) + [-1], dtype=int)


async def _drain(run: PipelineRun) -> list:
    return [event async for event in run.run()]


def _make_run(
    settings: Settings,
    papers: list[Paper],
    monkeypatch: pytest.MonkeyPatch,
    **overrides,
) -> PipelineRun:
    async def fake_enrich(paper_list, _settings, **kwargs):
        return list(paper_list), []

    def fake_rank(topic, paper_list, _settings, **kwargs):
        return _ranked(list(paper_list))

    async def fake_extract(paper_list, _settings, **kwargs):
        out = {}
        for paper in paper_list:
            out[paper.paper_id] = PaperExtraction(
                problem="Study test topic.",
                method="Controlled experiment.",
                results=f"Result for {paper.paper_id}.",
                contribution="A test contribution.",
                evidence={"results": paper.abstract[:40]},
                status="ok",
            )
        from pipeline.extract import Coverage

        return out, Coverage(ok=len(out), failed=0, skipped=0)

    monkeypatch.setattr(
        "pipeline.stages.retrieve_mod.fetch_candidates", lambda *a, **k: list(papers)
    )
    monkeypatch.setattr(stages_mod.enrich_mod, "enrich_papers", fake_enrich)
    monkeypatch.setattr(stages_mod.rerank_mod, "rank_papers", fake_rank)
    monkeypatch.setattr(stages_mod.extract_mod, "extract_all", fake_extract)
    params = {
        "arxiv_client": object(),
        "scorer": object(),
        "embedder": _FakeEmbedder(),
        "reducer": None,
        "clusterer": _FakeClusterer(),
        "s2": None,
        "openalex": None,
    }
    params.update(overrides)
    return PipelineRun("test topic", settings, completer=None, **params)


def test_full_run_reaches_ready_with_degraded_false(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    papers = _papers()
    run = _make_run(settings, papers, monkeypatch)
    events = asyncio.run(_drain(run))
    stages_done = {e.stage for e in events if e.status == "done"}
    assert {
        "retrieval",
        "enrichment",
        "rerank",
        "extraction",
        "layout",
        "synthesis",
    } <= stages_done
    assert run.state.degraded is False
    assert all(e.degraded is False for e in events)
    with store.session(settings) as conn:
        row = store.fetch_landscape(conn, run.state.landscape_id or 0)
    assert row is not None and row["status"] == "ready"
    assert row["narrative_status"] in ("ok", "partial", "fallback")


def test_enrichment_failure_still_reaches_synthesis_degraded(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    papers = _papers()
    run = _make_run(settings, papers, monkeypatch)

    async def boom(paper_list, _settings, **kwargs):
        raise ConnectionError("S2 blackholed")

    monkeypatch.setattr(stages_mod.enrich_mod, "enrich_papers", boom)
    events = asyncio.run(_drain(run))
    stages_done = {e.stage for e in events if e.status == "done"}
    assert "synthesis" in stages_done
    assert run.state.degraded is True
    assert any(e.degraded is True for e in events)
    with store.session(settings) as conn:
        row = store.fetch_landscape(conn, run.state.landscape_id or 0)
    assert row is not None and row["status"] == "ready"


def test_layout_failure_marks_landscape_failed_with_retryable(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    papers = _papers()
    run = _make_run(settings, papers, monkeypatch)

    class BoomClusterer:
        def fit_predict(self, _matrix: np.ndarray) -> np.ndarray:
            raise RuntimeError("UMAP exploded")

    run._clusterer = BoomClusterer()
    events = asyncio.run(_drain(run))
    errors = [e for e in events if e.status == "error"]
    assert errors and errors[0].stage == "layout"
    assert errors[0].payload.get("retryable") is False
    with store.session(settings) as conn:
        row = store.fetch_landscape(conn, run.state.landscape_id or 0)
    assert row is not None and row["status"] == "failed"
