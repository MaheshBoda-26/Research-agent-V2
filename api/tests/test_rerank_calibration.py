"""Tests for reranking: calibration, blending, and the degradation ladder.

No model is ever loaded. The cross-encoder is injected as a fake through the
``scorer=`` seam and the judge through the ``judge=`` seam, so the whole
stage is exercised offline and in milliseconds (§11.2).

The calibration tests are the important ones. They encode the rule that scores
are absolute: a paper's score must not depend on what else happened to be in the
batch, because a batch-relative score silently makes every retrieved set look
equally good.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from conftest import make_paper
from pydantic import ValidationError

from config import Settings
from models import JudgeBatch, JudgeScore
from pipeline.rerank import (
    DEGRADED_SCORE_CAP,
    EncoderScores,
    calibrate,
    judge_papers,
    parse_judge_scores,
    rank_papers,
    select_top,
)


class FakeEncoder:
    """Returns preset calibrated scores (and optionally raw logits), or raises."""

    def __init__(
        self,
        scores: list[float] | None = None,
        logits: list[float] | None = None,
        raises: Exception | None = None,
    ):
        self._scores = scores
        self._logits = logits
        self._raises = raises
        self.calls: list[list[str]] = []

    def score(self, query: str, texts) -> EncoderScores:
        self.calls.append(list(texts))
        if self._raises is not None:
            raise self._raises
        calibrated = self._scores if self._scores is not None else [5.0] * len(texts)
        # Default raw mirrors the calibrated values: any monotonic stand-in is
        # sufficient because relative_scores only uses ordering.
        raw = self._logits if self._logits is not None else list(calibrated)
        return EncoderScores(raw=list(raw), calibrated=list(calibrated))


class FakeJudge:
    """Returns a JudgeBatch built by a callback, or raises."""

    def __init__(self, callback=None, raises: Exception | None = None):
        self._callback = callback
        self._raises = raises
        self.calls = 0
        self.last_user_prompt = ""

    @property
    def model_name(self) -> str:
        return "fake-judge"

    def complete_json(self, *, system, user, schema, stage="", **kwargs):
        self.calls += 1
        self.last_user_prompt = user
        if self._raises is not None:
            raise self._raises
        if self._callback is None:
            return JudgeBatch(scores=[])
        return self._callback(user)


def papers(n: int) -> list:
    return [make_paper(paper_id=f"p{i:03d}", title=f"Title {i}", abstract=f"Abstract {i}") for i in range(n)]


def _batch(*pairs) -> JudgeBatch:
    return JudgeBatch(scores=[JudgeScore(paper_id=pid, relevance_0_10=s) for pid, s in pairs])


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #


def test_sigmoid_midpoint_is_neutral():
    assert calibrate([0.0])[0] == pytest.approx(5.0)


def test_large_positive_logit_saturates_high():
    assert calibrate([10.0])[0] > 9.9


def test_large_negative_logit_saturates_low():
    assert calibrate([-10.0])[0] < 0.1


def test_calibrate_is_monotonic():
    scores = calibrate([-8.0, -2.0, 0.0, 2.0, 8.0])
    assert scores == sorted(scores)


def test_calibrate_handles_extreme_logits_without_overflow():
    """Naive 1/(1+exp(-x)) overflows for very negative x."""
    assert calibrate([-1000.0])[0] == pytest.approx(0.0)
    assert calibrate([1000.0])[0] == pytest.approx(10.0)


def test_scores_are_absolute_and_not_batch_normalized():
    """A lone irrelevant paper must NOT be rescaled up to 10.

    This is the anti-min-max regression test: with batch normalization a single
    candidate always scores 10, however bad it is.
    """
    settings = replace(Settings.from_env(), rerank_seed_count=0)
    encoder = FakeEncoder(scores=[0.0001])
    ranked = rank_papers("topic", [make_paper()], settings, scorer=encoder)
    assert ranked[0].relevance_score == pytest.approx(0.0001, abs=1e-3)
    assert ranked[0].relevance_score < 1.0


def test_scores_do_not_depend_on_batchmates():
    """The same logit scores the same wherever it appears in a batch."""
    settings = replace(Settings.from_env(), rerank_seed_count=0)
    alone = rank_papers("topic", [make_paper(paper_id="solo")], settings, scorer=FakeEncoder([3.0]))
    crowded = rank_papers(
        "topic",
        [make_paper(paper_id="solo"), make_paper(paper_id="other")],
        settings,
        scorer=FakeEncoder([3.0, 9.0]),
    )
    solo = next(item for item in crowded if item.paper.paper_id == "solo")
    assert solo.relevance_score == pytest.approx(alone[0].relevance_score)


# --------------------------------------------------------------------------- #
# Degradation ladder
# --------------------------------------------------------------------------- #


def test_empty_input_returns_empty():
    settings = replace(Settings.from_env(), rerank_seed_count=0)
    assert rank_papers("topic", [], settings, scorer=FakeEncoder([])) == []


def test_encoder_failure_preserves_retrieval_order_with_capped_scores():
    settings = replace(Settings.from_env(), rerank_seed_count=0)
    encoder = FakeEncoder(raises=RuntimeError("model missing"))
    ranked = rank_papers("topic", papers(3), settings, scorer=encoder)
    assert [r.paper.paper_id for r in ranked] == ["p000", "p001", "p002"]
    assert [r.rank for r in ranked] == [1, 2, 3]
    assert all(r.relevance_score == DEGRADED_SCORE_CAP for r in ranked)
    assert all(r.rerank_source == "fusion-fallback" for r in ranked)
    assert all(r.cross_encoder_logit is None for r in ranked)


def test_count_mismatch_is_treated_as_encoder_failure():
    """A scorer returning the wrong number of values cannot be trusted at all."""
    settings = replace(Settings.from_env(), rerank_seed_count=0)
    ranked = rank_papers("topic", papers(3), settings, scorer=FakeEncoder([1.0, 2.0]))
    assert all(r.rerank_source == "fusion-fallback" for r in ranked)
    assert [r.rank for r in ranked] == [1, 2, 3]


def test_judge_failure_leaves_cross_encoder_scores():
    """The judge is optional signal: its failure costs the blend, not the stage."""
    settings = replace(Settings.from_env(), rerank_seed_count=2)
    encoder = FakeEncoder([9.0, 8.0])
    judge = FakeJudge(raises=RuntimeError("LLM down"))
    ranked = rank_papers("topic", papers(2), settings, scorer=encoder, judge=judge)
    assert {r.rerank_source for r in ranked} == {"cross-encoder"}
    assert ranked[0].paper.paper_id == "p000"
    assert ranked[0].relevance_score == pytest.approx(9.0)


def test_no_judge_means_cross_encoder_source_and_ce_scores():
    settings = replace(Settings.from_env(), rerank_seed_count=2)
    ranked = rank_papers("topic", papers(2), settings, scorer=FakeEncoder([4.0, 7.0]), judge=None)
    assert [r.paper.paper_id for r in ranked] == ["p001", "p000"]
    assert [r.rank for r in ranked] == [1, 2]
    assert all(r.rerank_source == "cross-encoder" for r in ranked)
    assert all(r.judge_score is None for r in ranked)


def test_blend_combines_ce_judge_and_citation_prior():
    """The full V2 blend: w_ce*ce + w_llm*judge + w_cite*prior."""
    settings = replace(Settings.from_env(), rerank_seed_count=2)
    encoder = FakeEncoder([8.0, 6.0])
    batch = [
        make_paper(paper_id="p000", citation_count=100),
        make_paper(paper_id="p001", citation_count=100),
    ]
    judge = FakeJudge(
        callback=lambda _: _batch(("p000", 4.0), ("p001", 4.0)),
    )
    ranked = rank_papers("topic", batch, settings, scorer=encoder, judge=judge)
    # max_count=100 → prior = log1p(100)/log1p(100)*10 = 10.
    # weights: citation 0.15; ce = 0.85*0.6 = 0.51; llm = 0.85*0.4 = 0.34.
    assert ranked[0].rerank_source == "blend"
    assert ranked[0].relevance_score == pytest.approx(0.51 * 8.0 + 0.34 * 4.0 + 0.15 * 10.0)
    assert ranked[0].judge_score == pytest.approx(4.0)
    assert ranked[0].citation_prior == pytest.approx(10.0)


def test_judge_only_scores_the_seed_slice():
    """Beyond the seed count the cross-encoder ordering is reliable enough."""
    settings = replace(Settings.from_env(), rerank_seed_count=2, rerank_judge_batch_size=10)
    encoder = FakeEncoder([9.0, 8.0, 7.0, 6.0, 5.0])
    judge = FakeJudge(callback=lambda _: _batch(("p000", 5.0), ("p001", 5.0)))
    ranked = rank_papers("topic", papers(5), settings, scorer=encoder, judge=judge)

    assert judge.calls == 1
    blended = [r for r in ranked if r.rerank_source == "blend"]
    assert len(blended) == 2
    # The two highest cross-encoder scores are the ones that got judged.
    assert {r.paper.paper_id for r in blended} == {"p000", "p001"}


def test_batching_sends_separate_calls():
    settings = replace(Settings.from_env(), rerank_seed_count=4, rerank_judge_batch_size=2)
    encoder = FakeEncoder([9.0, 8.0, 7.0, 6.0])
    batch = papers(4)
    by_id = {paper.paper_id: paper for paper in batch}

    def callback(user: str):
        ids = [line.split("]")[0][1:] for line in user.splitlines() if line.startswith("[")]
        return JudgeBatch(scores=[JudgeScore(paper_id=pid, relevance_0_10=5.0) for pid in ids])

    judge = FakeJudge(callback=callback)
    ranked = rank_papers("topic", batch, settings, scorer=encoder, judge=judge)
    assert judge.calls == 2
    assert all(r.rerank_source == "blend" for r in ranked)
    assert by_id  # papers were addressable by id throughout


def test_ranks_are_dense_and_descending():
    settings = replace(Settings.from_env(), rerank_seed_count=0)
    encoder = FakeEncoder([1.0, 9.0, 5.0])
    ranked = rank_papers("topic", papers(3), settings, scorer=encoder, judge=None)
    assert [r.rank for r in ranked] == [1, 2, 3]
    assert [r.relevance_score for r in ranked] == sorted([r.relevance_score for r in ranked], reverse=True)


def test_progress_events_are_emitted():
    settings = replace(Settings.from_env(), rerank_seed_count=2)
    events: list[dict] = []
    rank_papers(
        "topic",
        papers(2),
        settings,
        scorer=FakeEncoder([9.0, 8.0]),
        judge=FakeJudge(callback=lambda _: _batch(("p000", 5.0), ("p001", 5.0))),
        progress=events.append,
    )
    assert {"event": "judge_start", "count": 2} in events
    assert {"event": "rerank_done", "ranked": 2} in events


def test_select_top_respects_limit():
    settings = replace(Settings.from_env(), rerank_seed_count=0)
    ranked = rank_papers("topic", papers(3), settings, scorer=FakeEncoder([3.0, 1.0, 2.0]))
    top = select_top(ranked, 2)
    assert [r.rank for r in top] == [1, 2]
    assert select_top(ranked, 0) == []
    assert select_top(ranked, 99) == ranked


# --------------------------------------------------------------------------- #
# Judge response validation
# --------------------------------------------------------------------------- #


def test_parse_accepts_a_perfectly_aligned_response():
    parsed = parse_judge_scores(_batch(("a", 8.0), ("b", 2.0)), {"a", "b"})
    assert parsed == {"a": 8.0, "b": 2.0}


def test_parse_rejects_a_missing_id():
    assert parse_judge_scores(_batch(("a", 8.0)), {"a", "b"}) is None


def test_parse_tolerates_a_fabricated_id_on_a_complete_response():
    """An invented id is dropped, not scored — the aligned part still counts."""
    batch = _batch(("a", 8.0), ("b", 2.0), ("zzz", 9.9))
    assert parse_judge_scores(batch, {"a", "b"}) == {"a": 8.0, "b": 2.0}


def test_parse_rejects_a_duplicate_id():
    assert parse_judge_scores(_batch(("a", 8.0), ("a", 2.0)), {"a", "b"}) is None


def test_parse_rejects_a_none_response():
    assert parse_judge_scores(None, {"a"}) is None


def test_parse_rejects_an_empty_response():
    assert parse_judge_scores(JudgeBatch(scores=[]), {"a"}) is None


def test_out_of_range_scores_are_rejected_by_the_schema():
    """Pydantic enforces 0-10 at the boundary, so a rogue score never arrives."""
    with pytest.raises(ValidationError):
        JudgeScore(paper_id="a", relevance_0_10=42.0)


def test_misaligned_batch_is_discarded_not_partially_applied():
    settings = replace(Settings.from_env(), rerank_seed_count=2, rerank_judge_batch_size=10)
    encoder = FakeEncoder([9.0, 8.0])
    # Only one score for two papers: a shifted score would corrupt the ranking.
    judge = FakeJudge(callback=lambda _: _batch(("p000", 10.0)))
    ranked = rank_papers("topic", papers(2), settings, scorer=encoder, judge=judge)
    assert {r.rerank_source for r in ranked} == {"cross-encoder"}


def test_judge_papers_without_a_completer_returns_nothing(settings: Settings):
    assert judge_papers("topic", papers(2), None, settings) == {}

    assert calibrate([-10.0])[0] < 0.1
