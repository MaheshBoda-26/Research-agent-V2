"""Tests for Task 7.4 — synthesis call A (the narrative) with its fallback (D1).

Pinned behaviours (plan Task 7.4):

* a valid model narrative (good title, >=3-sentence summary) is used -> ``ok``;
* a malformed narrative (``None`` from the completer) -> ``fallback`` with a
  **non-empty** deterministic summary;
* a title that restates the topic (D4) is rejected -> ``fallback``;
* a summary under three sentences is rejected -> ``fallback``;
* **no code path yields an empty summary with ``status='ok'``**.
"""

from __future__ import annotations

from conftest import make_paper

from config import Settings
from models import Claims, Narrative, OpenProblem, Paper, ReadingPath, ReadingStep, Tension
from pipeline.synthesize import synthesize

TOPIC = "mixture of experts routing"
#: A title that shares <0.5 token overlap with TOPIC (passes the D4 gate).
GOOD_TITLE = "Adaptive Routing Strategies for Sparse Models"
#: A 3-sentence summary (the B.6 minimum).
GOOD_SUMMARY = (
    "This landscape covers sparse model routing. "
    "Papers group into load-balancing and expert-specialization areas. "
    "Several open questions remain about scaling."
)


class StageCompleter:
    """Routes a canned response per synthesis stage (offline JSONCompleter, §11.2).

    ``responses`` maps stage -> a Pydantic instance, ``None`` (malformed /
    refused), or an ``Exception`` (transport failure).
    """

    def __init__(self, responses: dict[str, object]):
        self.responses = responses
        self.calls: list[str] = []

    def complete_json(self, *, system, user, schema, stage="", **kwargs):
        self.calls.append(stage)
        resp = self.responses.get(stage)
        if isinstance(resp, Exception):
            raise resp
        return resp

    def complete_text(self, *, system, user):
        return None


def _papers(count: int = 6) -> list[Paper]:
    return [
        make_paper(paper_id=f"p{i}", title=f"Routing technique number {i}")
        for i in range(count)
    ]


def _clusters() -> list[dict]:
    return [
        {"local_label": 0, "label": "Load Balancing", "paper_ids": ["p0", "p1", "p2"]},
        {"local_label": 1, "label": "Expert Specialization", "paper_ids": ["p3", "p4", "p5"]},
    ]


def _full_success() -> StageCompleter:
    """A completer whose three calls all return valid envelopes."""
    ids = [f"p{i}" for i in range(6)]
    return StageCompleter(
        {
            "synthesis-narrative": Narrative(title=GOOD_TITLE, summary=GOOD_SUMMARY),
            "synthesis-claims": Claims(
                tensions=[
                    Tension(
                        statement="p0 and p1 disagree on load imbalance.",
                        paper_a_id="p0",
                        paper_b_id="p1",
                    )
                ],
                open_problems=[
                    OpenProblem(
                        statement="No work scales routing beyond 1k experts.",
                        why_open="All supplied papers cap at 256 experts.",
                        supporting_paper_ids=["p2", "p3"],
                    )
                ],
            ),
            "synthesis-reading-path": ReadingPath(
                steps=[
                    ReadingStep(paper_id=pid, position=i + 1, why="foundational")
                    for i, pid in enumerate(ids[:5])
                ]
            ),
        }
    )


def test_valid_narrative_is_used_and_status_ok(settings: Settings):
    out = synthesize(
        _clusters(), _papers(), {}, [], TOPIC, settings, completer=_full_success()
    )
    assert out["title"] == GOOD_TITLE
    assert out["summary"] == GOOD_SUMMARY
    assert out["narrative_status"] == "ok"


def test_malformed_narrative_falls_back_with_nonempty_summary(settings: Settings):
    completer = _full_success()
    completer.responses["synthesis-narrative"] = None  # malformed JSON -> None
    out = synthesize(_clusters(), _papers(), {}, [], TOPIC, settings, completer=completer)
    assert out["narrative_status"] == "fallback"
    assert out["summary"].strip()  # never empty
    assert out["title"].strip()
    assert TOPIC in out["summary"]  # deterministic template mentions the topic


def test_title_restating_topic_is_rejected(settings: Settings):
    completer = _full_success()
    # "Mixture of Experts Routing" restates the topic — the D4 gate must reject it.
    completer.responses["synthesis-narrative"] = Narrative(
        title="Mixture of Experts Routing", summary=GOOD_SUMMARY
    )
    out = synthesize(_clusters(), _papers(), {}, [], TOPIC, settings, completer=completer)
    assert out["narrative_status"] == "fallback"
    assert out["title"] != "Mixture of Experts Routing"


def test_thin_summary_is_rejected(settings: Settings):
    completer = _full_success()
    completer.responses["synthesis-narrative"] = Narrative(
        title=GOOD_TITLE, summary="Too short."  # 1 sentence < the 3-sentence minimum
    )
    out = synthesize(_clusters(), _papers(), {}, [], TOPIC, settings, completer=completer)
    assert out["narrative_status"] == "fallback"


def test_no_completer_falls_back(settings: Settings):
    out = synthesize(_clusters(), _papers(), {}, [], TOPIC, settings, completer=None)
    assert out["narrative_status"] == "fallback"
    assert out["summary"].strip()


def test_no_empty_summary_with_ok_status(settings: Settings):
    """Invariant: an empty summary can never be reported with status 'ok'."""
    for narrative in (
        Narrative(title=GOOD_TITLE, summary=""),       # empty summary
        Narrative(title="", summary=GOOD_SUMMARY),     # empty title
        None,                                          # malformed
    ):
        completer = _full_success()
        completer.responses["synthesis-narrative"] = narrative
        out = synthesize(
            _clusters(), _papers(), {}, [], TOPIC, settings, completer=completer
        )
        if out["narrative_status"] == "ok":
            assert out["summary"].strip()


# --------------------------------------------------------------------------- #
# The narrative prompt itself (content, not just the canned response)
# --------------------------------------------------------------------------- #


def test_narrative_prompt_includes_every_cluster():
    """Regression: the return must sit outside the cluster loop (all clusters)."""
    from prompts.synthesize import build_narrative_prompt

    clusters = [
        {"local_label": 0, "label": "Load Balancing", "paper_count": 3},
        {"local_label": 1, "label": "Expert Specialization", "paper_count": 4},
        {"local_label": 2, "label": "Routing Objectives", "paper_count": 2},
    ]
    prompt = build_narrative_prompt(TOPIC, clusters, {0: ["t0"], 1: ["t1"], 2: ["t2"]})
    for label in (0, 1, 2):
        assert f"Cluster {label}" in prompt

