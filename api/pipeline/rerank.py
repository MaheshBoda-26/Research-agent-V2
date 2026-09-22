"""Stage 3 — reranking (Phase 4).

Ported from V1's ``pipeline/rerank.py`` with the V2 blend added. Two passes:

1. **Cross-encoder** over every candidate. Local, fast, free. Raw logits are
   mapped to 0-10 through the model's own sigmoid, which for ms-marco
   cross-encoders is the trained relevance probability.
2. **LLM judge** over the top ``RERANK_SEED_COUNT`` only, in batches. Slower
   and costs tokens, so it is spent where it changes the outcome.

V2 adds a **citation prior** (Task 4.3): ``log1p(count)/log1p(max_count)*10``,
log-scaled so an 18k-citation survey does not swamp a 40-citation frontier
paper. Papers with an unknown count get the batch's *median* prior — absence
of data is not evidence of irrelevance, and must never read as zero.

Calibration is the part that is easy to get wrong. Scores are **absolute**:
``sigmoid(logit) * 10`` is query- and batch-independent. Min-max normalising
inside a batch would force the best candidate to 10 even when every candidate
is irrelevant (§2.10) — never do it.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from statistics import median
from typing import Any, Callable, Protocol, Sequence

from config import Settings
from llm.protocol import JSONCompleter
from models import JudgeBatch, Paper, RankedPaper
from prompts.rerank import JUDGE_SYSTEM_PROMPT, build_judge_user_prompt

#: Single-dict progress events, e.g. ``{"event": "judge_start", "count": n}``.
ProgressCallback = Callable[[dict], None]

logger = logging.getLogger(__name__)

#: Ceiling for a ranking that carries no absolute relevance signal.
DEGRADED_SCORE_CAP = 5.0

#: Default cross-encoder. Fast, tiny (~90 MB), adequate for the top-40
#: correction the judge then applies.
DEFAULT_CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _sigmoid(x: float) -> float:
    """Numerically stable logistic sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


@dataclass(frozen=True)
class EncoderScores:
    """Raw logits (ordering) and calibrated 0-10 values (magnitude)."""

    raw: list[float]
    calibrated: list[float]


class CrossEncoderLike(Protocol):
    """What ``rank_papers`` needs from a scorer — the test seam (§11.2)."""

    def score(self, query: str, texts: Sequence[str]) -> EncoderScores: ...


class CrossEncoderReranker:
    """Lazy wrapper around a sentence-transformers cross-encoder."""

    def __init__(
        self,
        model_name: str = DEFAULT_CROSS_ENCODER,
        device: str = "cpu",
        max_length: int = 512,
        batch_size: int = 16,
    ) -> None:
        self.model_name = model_name
        self.device = device
        # 512, not 256: abstracts run 1200-1900 chars (~300-470 tokens) and a
        # smaller window truncates the tail, where results/contribution live.
        self.max_length = max_length
        self.batch_size = batch_size
        self._model: Any = None

    def _load_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            # Installed 3.3.1 API spells it ``max_length`` (docs/decisions.md,
            # Task 4.0). Bumping the pin to v6 renames it to ``max_seq_length``.
            self._model = CrossEncoder(self.model_name, device=self.device, max_length=self.max_length)
            logger.info("Loaded cross-encoder %s on %s", self.model_name, self.device)
        return self._model

    def score(self, query: str, texts: Sequence[str]) -> EncoderScores:
        """Return raw logits and calibrated 0-10 relevance, in input order."""
        if not texts:
            return EncoderScores(raw=[], calibrated=[])
        model = self._load_model()
        pairs = [(query, text) for text in texts]
        raw = model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        logits = [float(score) for score in raw]
        return EncoderScores(raw=logits, calibrated=calibrate(logits))


def calibrate(raw_scores: Sequence[float]) -> list[float]:
    """Map raw logits to an absolute 0-10 scale via the model-native sigmoid.

    Deliberately not min-max: the result is query- and batch-independent.
    """
    return [round(10.0 * _sigmoid(float(score)), 4) for score in raw_scores]


def relative_scores(logits: Sequence[float]) -> list[float]:
    """Percentile-rank the logits onto 0-10 for display.

    RELATIVE BY CONSTRUCTION: a 10.0 means "best in this landscape", not
    "highly relevant" — a different claim from ``relevance_score``, never to be
    used for a threshold. Computed on read, never stored (V1's rationale: a
    stored percentile goes stale as the landscape grows). Ties share the
    average rank.
    """
    count = len(logits)
    if count == 0:
        return []
    if count == 1:
        return [5.0]

    order = sorted(range(count), key=lambda index: logits[index])
    ranks = [0.0] * count
    start = 0
    while start < count:
        end = start
        while end + 1 < count and logits[order[end + 1]] == logits[order[start]]:
            end += 1
        average_rank = (start + end) / 2.0
        for position in range(start, end + 1):
            ranks[order[position]] = average_rank
        start = end + 1

    return [round(10.0 * rank / (count - 1), 4) for rank in ranks]


#: Process-wide default reranker, so the model loads once per server.
_reranker: CrossEncoderReranker | None = None


def get_reranker(settings: Settings) -> CrossEncoderReranker:
    """Return the shared reranker, rebuilding it if the configuration changed."""
    global _reranker
    if _reranker is None or (
        _reranker.model_name != settings.cross_encoder_model
        or _reranker.device != settings.cross_encoder_device
        or _reranker.max_length != settings.cross_encoder_max_length
    ):
        _reranker = CrossEncoderReranker(
            model_name=settings.cross_encoder_model,
            device=settings.cross_encoder_device,
            max_length=settings.cross_encoder_max_length,
            batch_size=settings.cross_encoder_batch_size,
        )
    return _reranker


def citation_prior(count: int | None, max_count: int) -> float | None:
    """Log-scaled 0-10 citation importance (Task 4.3).

    ``None`` propagates: the caller imputes the batch median, because an
    unknown count must not read as zero (anti-pattern ledger §2.10).
    """
    if count is None:
        return None
    if max_count <= 0:
        return 0.0
    return round(math.log1p(count) / math.log1p(max_count) * 10.0, 4)


def parse_judge_scores(result: JudgeBatch | None, expected_ids: set[str]) -> dict[str, float] | None:
    """Validate a judge response against the ids actually sent.

    B.2 alignment contract — return ``None`` (discard the whole batch) when the
    response is missing an id, duplicates one, or is absent entirely. A judge
    inventing an id that was never supplied gets that entry *dropped*, not
    scored: a fabricated id would attach a score to a paper the judge never
    saw. But an invented id *on top of* a complete response is tolerated — the
    complete response is still aligned.

    Scores outside 0-10 cannot reach here; the pydantic schema rejects them.
    """
    if result is None:
        return None
    parsed: dict[str, float] = {}
    for entry in result.scores:
        if entry.paper_id not in expected_ids:
            continue  # fabricated id — dropped, never scored
        if entry.paper_id in parsed:
            return None  # duplicate — ambiguous, discard the batch
        parsed[entry.paper_id] = entry.relevance_0_10
    if set(parsed) != expected_ids:
        return None  # missing id — a shifted score would corrupt the ranking
    return parsed


def judge_papers(
    topic: str,
    papers: list[Paper],
    completer: JSONCompleter | None,
    settings: Settings,
) -> dict[str, float]:
    """LLM-judge a slice of papers; return ``paper_id -> 0-10``.

    A failed or misaligned batch is simply absent, which the caller treats as
    "no LLM signal for this paper" rather than a zero.
    """
    if completer is None or not papers:
        return {}

    results: dict[str, float] = {}
    batch_size = max(1, settings.rerank_judge_batch_size)

    for offset in range(0, len(papers), batch_size):
        batch = papers[offset : offset + batch_size]
        try:
            judged = completer.complete_json(
                system=JUDGE_SYSTEM_PROMPT,
                user=build_judge_user_prompt(topic, batch),
                schema=JudgeBatch,
                stage="rerank-judge",
                temperature=0.0,
            )
        except Exception as exc:  # noqa: BLE001 - the judge is an optional signal
            logger.warning("Judge batch failed: %s", exc)
            continue

        parsed = parse_judge_scores(
            judged if isinstance(judged, JudgeBatch) else None,
            {paper.paper_id for paper in batch},
        )
        if parsed is None:
            logger.warning(
                "Discarding judge batch: response did not align with the ids sent",
            )
            continue
        results.update(parsed)

    return results


def rank_papers(
    topic: str,
    papers: list[Paper],
    settings: Settings,
    *,
    scorer: CrossEncoderLike | None = None,
    judge: JSONCompleter | None = None,
    progress: ProgressCallback | None = None,
) -> list[RankedPaper]:
    """Rerank ``papers`` and return them best-first with ``rank`` populated.

    Signature per Appendix A.6. ``scorer`` and ``judge`` are the injected
    seams; ``None`` means "use the real cross-encoder" / "no judge available".

    Degradation ladder, in order of preference:

    ``blend``                  cross-encoder, judge and citations all contributed
    ``cross-encoder+citation`` no judge signal (judge off or failed), citations ran
    ``cross-encoder``          same, but citations are unknown for the corpus
    ``fusion-fallback``        the cross-encoder failed; retrieval order kept,
                               scores capped at ``DEGRADED_SCORE_CAP`` because
                               rank order carries no absolute relevance signal
    """
    if not papers:
        return []

    scores: EncoderScores | None = None
    try:
        active_scorer = scorer if scorer is not None else get_reranker(settings)
        scores = active_scorer.score(topic, [paper.rerank_text for paper in papers])
    except Exception as exc:  # noqa: BLE001 - degrade to retrieval order
        logger.warning(
            "Cross-encoder failed (%s); falling back to capped retrieval order",
            exc,
        )

    counts_match = scores is not None and len(scores.calibrated) == len(papers) and len(scores.raw) == len(papers)
    if not counts_match:
        if scores is not None:
            logger.warning(
                "Cross-encoder returned %d calibrated / %d raw scores for %d papers; " "ignoring them",
                len(scores.calibrated),
                len(scores.raw),
                len(papers),
            )
        return _fusion_fallback(papers)

    assert scores is not None  # narrowed by counts_match
    ce_scores = scores.calibrated
    logits = scores.raw

    # Citation priors (Task 4.3): an unknown count imputes the batch median.
    counts = [paper.citation_count for paper in papers]
    known = [count for count in counts if count is not None]
    max_count = max(known, default=0)
    if known:
        batch_median = median([citation_prior(c, max_count) or 0.0 for c in known])
    else:
        batch_median = 5.0
    priors = []
    for count in counts:
        prior = citation_prior(count, max_count)
        priors.append(prior if prior is not None else batch_median)
    citations_known = bool(known)

    # The judge only sees the top slice. Beyond it, the cross-encoder ordering is
    # reliable enough that spending tokens would not change the outcome.
    seed_count = max(0, settings.rerank_seed_count)
    ranked_by_ce = sorted(zip(papers, ce_scores, strict=True), key=lambda pair: pair[1], reverse=True)
    judge_targets = [paper for paper, _ in ranked_by_ce[:seed_count]]
    if judge_targets and progress is not None:
        progress({"event": "judge_start", "count": len(judge_targets)})
    judge_scores = judge_papers(topic, judge_targets, judge, settings)

    # Blend weights. The V2 three-way blend spends citation_blend_weight on the
    # prior; the rest keeps the V1 ce:llm ratio (which sums to 1.0 per config).
    weight_citation = settings.citation_blend_weight
    remaining = 1.0 - weight_citation
    weight_ce = remaining * settings.rerank_blend_ce
    weight_llm = remaining * settings.rerank_blend_llm

    ranked: list[RankedPaper] = []
    for index, (paper, ce_score) in enumerate(zip(papers, ce_scores, strict=True)):
        llm_score = judge_scores.get(paper.paper_id)
        prior = priors[index]

        if llm_score is not None:
            final = weight_ce * ce_score + weight_llm * llm_score
            final += weight_citation * prior
            source = "blend"
        elif citations_known:
            # No judge signal: renormalise the remaining weight onto what ran.
            cite_w = weight_citation / (weight_ce + weight_citation)
            final = (1.0 - cite_w) * ce_score + cite_w * prior
            source = "cross-encoder+citation"
        else:
            final = ce_score
            source = "cross-encoder"

        ranked.append(
            RankedPaper(
                paper=paper,
                rank=0,  # placeholder; dense ranks assigned after the sort below
                relevance_score=round(min(10.0, max(0.0, final)), 4),
                cross_encoder_logit=logits[index],
                citation_prior=prior,
                judge_score=llm_score,
                rerank_source=source,
            )
        )

    ranked.sort(key=lambda item: (-item.relevance_score, item.paper.paper_id))
    for index, item in enumerate(ranked, start=1):
        item.rank = index

    if progress is not None:
        progress({"event": "rerank_done", "ranked": len(ranked)})
    return ranked


def _fusion_fallback(papers: list[Paper]) -> list[RankedPaper]:
    """Cross-encoder unavailable: keep retrieval order, cap the scores."""
    return [
        RankedPaper(
            paper=paper,
            relevance_score=DEGRADED_SCORE_CAP,
            cross_encoder_logit=None,
            citation_prior=None,
            judge_score=None,
            rerank_source="fusion-fallback",
            rank=index,
        )
        for index, paper in enumerate(papers, start=1)
    ]


def select_top(ranked: list[RankedPaper], limit: int) -> list[RankedPaper]:
    """Take the top ``limit`` papers, preserving rank."""
    return ranked[: max(0, limit)]


__all__ = [
    "DEFAULT_CROSS_ENCODER",
    "DEGRADED_SCORE_CAP",
    "CrossEncoderLike",
    "CrossEncoderReranker",
    "EncoderScores",
    "calibrate",
    "citation_prior",
    "get_reranker",
    "judge_papers",
    "parse_judge_scores",
    "rank_papers",
    "relative_scores",
    "select_top",
]
