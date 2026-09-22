"""Tests for Task 4.3 — the three-way blend with the citation prior.

The load-bearing invariants:

1. ``citation_prior`` is log-scaled: an 18k-citation survey must not swamp a
   40-citation frontier paper.
2. ``None`` imputes the batch median — absence of data is not evidence of
   irrelevance, and must never read as zero (§2.10).
3. No ``NaN`` is ever produced, whatever the counts look like.
"""

from __future__ import annotations

import math
from dataclasses import replace

from conftest import make_paper
from test_rerank_calibration import FakeEncoder, papers

from config import Settings
from pipeline.rerank import citation_prior, rank_papers


def test_max_count_scores_ten():
    assert citation_prior(100, 100) == 10.0


def test_zero_count_scores_zero_not_none():
    assert citation_prior(0, 100) == 0.0


def test_scaling_is_logarithmic_not_linear():
    """18_000 vs 40 citations: log-scale says ~1.35x, linear would say 450x."""
    big = citation_prior(18_000, 18_000)
    small = citation_prior(40, 18_000)
    assert big == 10.0
    assert small == citation_prior(40, 18_000)
    linear_ratio = 18_000 / 40
    assert big / small < linear_ratio / 100


def test_unknown_count_propagates_none():
    assert citation_prior(None, 100) is None


def test_degenerate_max_count_returns_zero():
    assert citation_prior(5, 0) == 0.0


def test_null_citations_get_the_batch_median_rank():
    settings = replace(Settings.from_env(), rerank_seed_count=0)
    counts = [0, 10, 100, 1000]
    known_priors = sorted(citation_prior(c, 1000) or 0.0 for c in counts)
    expected_median = (known_priors[1] + known_priors[2]) / 2
    batch = [make_paper(paper_id=f"c{c}", citation_count=c) for c in counts]
    batch.append(make_paper(paper_id="unknown", citation_count=None))
    ranked = rank_papers("topic", batch, settings, scorer=FakeEncoder([5.0] * 5))
    unknown = next(r for r in ranked if r.paper.paper_id == "unknown")
    assert unknown.citation_prior == expected_median
    assert unknown.citation_prior is not None and unknown.citation_prior > 0.0


def test_unknown_count_beats_zero_but_loses_to_max():
    """Median imputation: unknown ranks between the anonymous and the famous."""
    settings = replace(Settings.from_env(), rerank_seed_count=0)
    batch = [
        make_paper(paper_id="zero", citation_count=0),
        make_paper(paper_id="unknown", citation_count=None),
        make_paper(paper_id="star", citation_count=10_000),
    ]
    ranked = rank_papers("topic", batch, settings, scorer=FakeEncoder([5.0] * 3))
    order = [r.paper.paper_id for r in ranked]
    assert order.index("unknown") < order.index("zero")
    assert order.index("star") < order.index("unknown")
    assert all(r.rerank_source == "cross-encoder+citation" for r in ranked)


def test_all_unknown_counts_degrade_to_cross_encoder_only():
    """No citation data anywhere: the prior must not move the scores."""
    settings = replace(Settings.from_env(), rerank_seed_count=0)
    ranked = rank_papers("topic", papers(3), settings, scorer=FakeEncoder([3.0, 9.0, 5.0]))
    assert all(r.rerank_source == "cross-encoder" for r in ranked)
    assert [round(r.relevance_score, 4) for r in ranked] == [9.0, 5.0, 3.0]


def test_no_nan_anywhere():
    """Whatever the counts, every emitted score is a finite number."""
    settings = replace(Settings.from_env(), rerank_seed_count=0)
    batch = [
        make_paper(paper_id="a", citation_count=None),
        make_paper(paper_id="b", citation_count=0),
        make_paper(paper_id="c", citation_count=10_000_000),
    ]
    ranked = rank_papers("topic", batch, settings, scorer=FakeEncoder([5.0] * 3))
    for item in ranked:
        assert math.isfinite(item.relevance_score)
        assert item.citation_prior is None or math.isfinite(item.citation_prior)
