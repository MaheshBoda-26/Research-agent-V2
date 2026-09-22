"""Tests for Task 4.4 — the LLM-as-judge cross-check (B.2).

What matters here:

(a) prompt blocks are tagged with each paper_id — the alignment contract;
(b) passages are truncated to 900 chars (the ≤6k-token budget);
(c) fabricated ids are dropped by ``parse_judge_scores``, never scored;
(d) a judge failure costs the blend, never the stage.
"""

from __future__ import annotations

from dataclasses import replace

from conftest import make_paper
from test_rerank_calibration import FakeEncoder, FakeJudge, _batch, papers

from config import Settings
from models import JudgeBatch, JudgeScore
from pipeline.rerank import judge_papers, rank_papers
from prompts.rerank import MAX_PASSAGE_CHARS, build_judge_user_prompt


def test_prompt_blocks_are_tagged_with_paper_id():
    batch = papers(3)
    prompt = build_judge_user_prompt("some topic", batch)
    assert "TOPIC: some topic" in prompt
    for paper in batch:
        assert f"[{paper.paper_id}]" in prompt


def test_prompt_truncates_passages_to_900_chars():
    paper = make_paper(paper_id="long", abstract="x" * 5000)
    prompt = build_judge_user_prompt("topic", [paper])
    assert len(prompt.split("[long] ", 1)[1]) <= MAX_PASSAGE_CHARS
    assert MAX_PASSAGE_CHARS == 900


def test_prompt_includes_title_and_abstract():
    paper = make_paper(paper_id="p", title="A very specific title", abstract="An abstract body")
    prompt = build_judge_user_prompt("topic", [paper])
    assert "A very specific title" in prompt
    assert "An abstract body" in prompt


def test_judge_scores_land_on_ranked_papers():
    settings = replace(Settings.from_env(), rerank_seed_count=2)
    encoder = FakeEncoder([9.0, 8.0])
    judge = FakeJudge(callback=lambda _: _batch(("p000", 1.0), ("p001", 9.0)))
    ranked = rank_papers("topic", papers(2), settings, scorer=encoder, judge=judge)
    by_id = {r.paper.paper_id: r for r in ranked}
    assert by_id["p000"].judge_score == 1.0
    assert by_id["p001"].judge_score == 9.0


def test_judge_can_overturn_cross_encoder_ordering():
    """The whole point of the judge: a CE-favourite demoted by a 1.0."""
    settings = replace(Settings.from_env(), rerank_seed_count=2)
    encoder = FakeEncoder([9.5, 9.4])
    judge = FakeJudge(callback=lambda _: _batch(("p000", 1.0), ("p001", 9.0)))
    ranked = rank_papers("topic", papers(2), settings, scorer=encoder, judge=judge)
    assert ranked[0].paper.paper_id == "p001"


def test_fabricated_id_is_dropped_not_scored():
    """A judge inventing an id on an otherwise complete batch: batch survives."""
    settings = replace(Settings.from_env(), rerank_seed_count=2, rerank_judge_batch_size=10)
    encoder = FakeEncoder([9.0, 8.0])
    payload = JudgeBatch(
        scores=[
            JudgeScore(paper_id="p000", relevance_0_10=7.0),
            JudgeScore(paper_id="p001", relevance_0_10=6.0),
            JudgeScore(paper_id="ghost-paper", relevance_0_10=10.0),
        ]
    )
    judge = FakeJudge(callback=lambda _: payload)
    ranked = rank_papers("topic", papers(2), settings, scorer=encoder, judge=judge)
    assert {r.rerank_source for r in ranked} == {"blend"}
    assert "ghost-paper" not in {r.paper.paper_id for r in ranked}


def test_judge_exception_costs_the_blend_not_the_stage():
    settings = replace(Settings.from_env(), rerank_seed_count=2)
    ranked = rank_papers(
        "topic",
        papers(2),
        settings,
        scorer=FakeEncoder([9.0, 8.0]),
        judge=FakeJudge(raises=RuntimeError("boom")),
    )
    assert [r.paper.paper_id for r in ranked] == ["p000", "p001"]
    assert all(r.judge_score is None for r in ranked)


def test_judge_papers_batches_and_skips_failures():
    settings = replace(Settings.from_env(), rerank_judge_batch_size=1, rerank_seed_count=3)
    calls: list[str] = []

    def callback(user: str):
        calls.append(user)
        if len(calls) == 2:
            raise RuntimeError("one bad batch")
        ids = [line.split("]")[0][1:] for line in user.splitlines() if line.startswith("[")]
        return JudgeBatch(scores=[JudgeScore(paper_id=pid, relevance_0_10=5.0) for pid in ids])

    class FlakyJudge(FakeJudge):
        def complete_json(self, **kwargs):
            self.calls += 1
            assert kwargs.get("stage") == "rerank-judge"
            return self._callback(kwargs["user"])

    judged = judge_papers("topic", papers(3), FlakyJudge(callback), settings)
    assert len(calls) == 3
    assert set(judged) == {"p000", "p002"}  # the failed batch is absent, not zero
