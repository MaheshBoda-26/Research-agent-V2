"""Tests for Task 4.5 — ``relative_scores``.

The percentile mapping is display-only: 10.0 means "best in this landscape",
not "highly relevant". These tests pin the mapping so nobody "fixes" it into
a threshold later. ``relative_scores`` takes raw logits (ordering is all it
needs); logit-less papers (``fusion-fallback``) carry no relative score at
all.
"""

from __future__ import annotations

from dataclasses import replace

from test_rerank_calibration import FakeEncoder, papers

from config import Settings
from pipeline.rerank import rank_papers, relative_scores


def test_empty_is_empty():
    assert relative_scores([]) == []


def test_single_paper_is_neutral():
    assert relative_scores([3.7]) == [5.0]


def test_best_is_ten_worst_is_zero():
    assert relative_scores([1.0, 2.0, 3.0]) == [0.0, 5.0, 10.0]


def test_monotonic_with_input_order():
    assert relative_scores([5.0, 1.0, 4.0, 2.0, 3.0]) == [10.0, 0.0, 7.5, 2.5, 5.0]


def test_ties_share_the_average_rank():
    assert relative_scores([1.0, 1.0, 3.0]) == [2.5, 2.5, 10.0]


def test_all_tied_lands_midscale_not_ten():
    """Three equally irrelevant papers must not all read as perfect."""
    assert relative_scores([2.0, 2.0, 2.0]) == [5.0, 5.0, 5.0]


def test_fusion_fallback_carries_no_scores_at_all():
    """Logit-less papers: relevance is capped and there is nothing relative."""
    settings = replace(Settings.from_env(), rerank_seed_count=0)
    ranked = rank_papers("topic", papers(2), settings, scorer=FakeEncoder(raises=RuntimeError("down")))
    assert all(r.cross_encoder_logit is None for r in ranked)
    assert all(r.judge_score is None for r in ranked)
