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
from typing import Any, Protocol, Sequence

from config import Settings
from llm.protocol import JSONCompleter
from models import JudgeBatch, Paper, RankedPaper
from prompts.rerank import JUDGE_SYSTEM_PROMPT, build_judge_user_prompt

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
            self._model = CrossEncoder(
                self.model_name, device=self.device, max_length=self.max_length
            )
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
