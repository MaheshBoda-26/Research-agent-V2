"""Tests for Task 6.3 — cluster naming with the anti-restatement gate (D4).

Three behaviours are pinned by the plan: the literal V1 failure — label
"Efficient Test-Time Scaling for LLMs" for that exact topic — must be
*rejected and replaced*; a genuinely good label must pass untouched; and any
label of two tokens or fewer must be rejected. The gate is deterministic, so
these tests are offline and exact.
"""

from __future__ import annotations

from config import Settings
from conftest import make_paper
from models import ClusterLabel, ClusterNaming, Paper
from pipeline.cluster import label_restates_topic, name_clusters
from prompts.cluster import CLUSTER_LABEL_SYSTEM_PROMPT, build_cluster_naming_prompt

#: The exact D4 failure, verbatim from the defect table.
D4_LABEL = "Efficient Test-Time Scaling for LLMs"
TOPIC = "efficient test-time scaling for LLMs"


class ScriptedCompleter:
    """Hands back one canned envelope — offline JSONCompleter (§11.2)."""

    def __init__(self, labels: list[ClusterLabel]) -> None:
        self.envelope = ClusterNaming(labels=labels)
        self.calls: list[dict] = []

    def complete_json(self, *, system, user, schema, stage="", **kwargs):
        self.calls.append({"system": system, "user": user, "stage": stage})
        return self.envelope

    def complete_text(self, *, system, user) -> str | None:
        return None


def _papers(count: int = 6) -> list[Paper]:
    """Distinctive titles so the tf-idf fallback has something to pick."""
    topics = ["Sparse Attention Kernels", "Mixture Depth Routing", "Curriculum Token Budgets"]
    return [
        make_paper(
            paper_id=f"p{i}",
            title=f"{topics[i % 3]} for Layer {i} of the model",
        )
        for i in range(count)
    ]


def _clusters(topic: str | None = TOPIC) -> list[dict]:
    return [
        {"local_label": -1, "size": 2, "topic": topic or "", "paper_ids": ["p0"]},
        {"local_label": 0, "size": 3, "topic": topic or "", "paper_ids": ["p0", "p1", "p2"]},
        {"local_label": 1, "size": 3, "topic": topic or "", "paper_ids": ["p3", "p4", "p5"]},
    ]


def test_d4_label_is_rejected_and_replaced(settings: Settings):
    """The literal V1 label must not survive the gate (plan B.4, Task 6.3)."""
    completer = ScriptedCompleter(
        [
            ClusterLabel(local_label=0, label=D4_LABEL, description="Restated the topic."),
            ClusterLabel(
                local_label=1,
                label="Adaptive Compute Allocation",
                description="Shared mechanism.",
            ),
        ]
    )
    named = name_clusters(_clusters(), _papers(), settings, completer=completer)
    by_label = {entry["local_label"]: entry for entry in named}

    assert by_label[0]["label"] != D4_LABEL
    # Replaced by the cluster's top tf-idf term, never by another restatement.
    assert label_restates_topic(by_label[0]["label"], TOPIC) is False
    assert by_label[0]["label"]  # non-empty
    # The good label passes through untouched.
    assert by_label[1]["label"] == "Adaptive Compute Allocation"
    assert by_label[1]["description"] == "Shared mechanism."


def test_unclustered_is_never_named(settings: Settings):
    named = name_clusters(_clusters(), _papers(), settings, completer=ScriptedCompleter([]))
    assert all(entry["local_label"] != -1 for entry in named)
    assert [entry["local_label"] for entry in named] == [0, 1]


def test_two_token_label_is_rejected(settings: Settings):
    """A label of <=2 tokens fails the gate even with zero topic overlap."""
    completer = ScriptedCompleter(
        [
            ClusterLabel(local_label=0, label="Token Thrift", description="Two words."),
            ClusterLabel(
                local_label=1,
                label="Adaptive Compute Allocation",
                description="Shared mechanism.",
            ),
        ]
    )
    named = name_clusters(_clusters(), _papers(), settings, completer=completer)
    by_label = {entry["local_label"]: entry for entry in named}
    assert by_label[0]["label"] != "Token Thrift"
    assert by_label[1]["label"] == "Adaptive Compute Allocation"


def test_good_label_passes_untouched(settings: Settings):
    completer = ScriptedCompleter(
        [
            ClusterLabel(
                local_label=0,
                label="Adaptive Compute Allocation",
                description="Routes extra compute to hard tokens.",
            ),
            ClusterLabel(
                local_label=1,
                label="Speculative Decoding Verification",
                description="Verifier-gated draft acceptance.",
            ),
        ]
    )
    named = name_clusters(_clusters(), _papers(), settings, completer=completer)
    assert [entry["label"] for entry in named] == [
        "Adaptive Compute Allocation",
        "Speculative Decoding Verification",
    ]


def test_missing_completer_still_yields_a_name_per_cluster(settings: Settings):
    named = name_clusters(_clusters(), _papers(), settings, completer=None)
    assert len(named) == 2
    assert all(entry["label"] for entry in named)
    assert all(entry["description"] for entry in named)


def test_unknown_local_labels_from_the_model_are_dropped(settings: Settings):
    completer = ScriptedCompleter(
        [
            ClusterLabel(local_label=99, label="Hallucinated Cluster", description="Nope."),
            ClusterLabel(local_label=-1, label="Should Not Appear", description="Nope."),
            ClusterLabel(local_label=0, label="Adaptive Compute Allocation", description="Kept."),
        ]
    )
    named = name_clusters(_clusters(), _papers(), settings, completer=completer)
    assert [entry["local_label"] for entry in named] == [0, 1]
    assert named[0]["label"] == "Adaptive Compute Allocation"
    # Cluster 1 got no model label, so it fell back to a deterministic name.
    assert named[1]["label"]


def test_empty_clusters_yield_no_names(settings: Settings):
    assert name_clusters([], [], settings, completer=None) == []
    only_unclustered = [{"local_label": -1, "paper_ids": [], "topic": TOPIC}]
    assert name_clusters(only_unclustered, [], settings, completer=None) == []


# --------------------------------------------------------------------------- #
# The gate itself (pure functions — no fixtures needed)
# --------------------------------------------------------------------------- #


def test_d4_label_restates_topic():
    assert label_restates_topic(D4_LABEL, TOPIC) is True


def test_mechanism_label_does_not_restate_topic():
    assert label_restates_topic("Adaptive Compute Allocation", TOPIC) is False


def test_partial_overlap_below_threshold_passes():
    # 2 of 5 label tokens are topic tokens (0.4) — under the 0.70 threshold.
    assert label_restates_topic("Attention Kernel Design", TOPIC) is False


def test_overlap_is_measured_on_the_label_side():
    # Every label token is in the topic (1.0) even though the topic is longer.
    assert label_restates_topic("Efficient Test-Time Scaling", TOPIC) is True


def test_threshold_is_configurable():
    assert label_restates_topic("Efficient Scaling", TOPIC, threshold=0.50) is True
    assert label_restates_topic("Efficient Scaling", TOPIC, threshold=1.10) is False


def test_empty_inputs_do_not_crash_the_gate():
    assert label_restates_topic("", TOPIC) is False
    assert label_restates_topic(D4_LABEL, "") is False


# --------------------------------------------------------------------------- #
# The prompt contract
# --------------------------------------------------------------------------- #


def test_prompt_carries_the_d4_negative_example_and_the_accepted_one():
    assert D4_LABEL in CLUSTER_LABEL_SYSTEM_PROMPT
    assert "Adaptive Compute Allocation" in CLUSTER_LABEL_SYSTEM_PROMPT
    assert "REJECTED" in CLUSTER_LABEL_SYSTEM_PROMPT
    assert "ACCEPTED" in CLUSTER_LABEL_SYSTEM_PROMPT


def test_prompt_requires_a_mechanism_or_axis():
    lowered = CLUSTER_LABEL_SYSTEM_PROMPT.lower()
    assert "mechanism" in lowered
    assert "axis" in lowered


def test_user_prompt_lists_clusters_with_five_exemplar_titles():
    titles = {0: [f"Paper number {i} about routing" for i in range(8)]}
    clusters = [{"local_label": 0, "size": 8, "paper_ids": []}]
    prompt = build_cluster_naming_prompt("mixture of experts", clusters, titles)
    assert "TOPIC: mixture of experts" in prompt
    assert "cluster 0 - 8 paper(s)" in prompt
    # Exactly five exemplars, as B.4 specifies — not all eight.
    assert prompt.count("  - ") == 5
    assert "Paper number 5" not in prompt


def test_user_prompt_collapses_and_truncates_long_titles():
    long_title = "  A   very   long   title   " + "x" * 400
    prompt = build_cluster_naming_prompt("t", [{"local_label": 0, "size": 1}], {0: [long_title]})
    assert "  - A very long title " in prompt  # whitespace collapsed
    assert "x" * 400 not in prompt  # truncated to MAX_TITLE_CHARS
    assert prompt.count("x") <= 200


