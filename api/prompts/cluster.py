"""B.4 — cluster naming prompt, with the D4 anti-restatement constraints.

One call labels every computed cluster from the topic plus up to five exemplar
titles per cluster (V1 made one call per cluster over each member's extracted
problem and contribution; the batch form keeps the prompt under the 2k-token
budget in B.4 and sees the same evidence the user does — titles).

Two constraints were added for V2, both because V1 shipped the defect recorded
as D4: the cluster for topic *"Efficient test-time scaling for LLMs"* was named
*"Efficient Test-Time Scaling for LLMs"*. First, the model must name a
**mechanism or axis**, not the topic rephrased, and the system prompt carries
worked examples — two rejected (one of them the literal D4 failure) and one
accepted. Second, prompts are suggestions: the deterministic half of the fix is
:func:`pipeline.cluster.label_restates_topic`, which rejects any label whose
token overlap with the topic reaches 0.70, or which is two tokens or fewer, and
replaces it with the cluster's top tf-idf term.
"""

from __future__ import annotations

#: B.4 hands the labeller five exemplar titles per cluster; longer titles are
#: truncated so the call stays inside its <=2k-token input budget.
MAX_EXEMPLARS = 5
MAX_TITLE_CHARS = 200

CLUSTER_LABEL_SYSTEM_PROMPT = """You name clusters of research papers on a topic map.

You are given a TOPIC and the computed clusters, each with an internal id and
up to five exemplar titles. Return JSON only, in exactly this shape:
{"labels": [{"local_label": 0, "label": "...", "description": "..."}]}

Rules for "label":
- 2 to 6 words, in the style of a research area name, not a sentence.
- Name the MECHANISM or axis the members share — "adaptive compute
  allocation", not "efficient test-time scaling".
- Must be distinct from the TOPIC and from every other label you return.
- Do not include the topic itself; the point is to distinguish this cluster
  from the other clusters.
- Title case. No trailing period.

Worked examples, for TOPIC "efficient test-time scaling for LLMs":
- "Efficient Test-Time Scaling for LLMs" — REJECTED: restates the topic verbatim.
- "Efficient Test-Time Scaling Methods" — REJECTED: the topic with one word swapped.
- "Adaptive Compute Allocation" — ACCEPTED: names the mechanism.

Rules for "description":
- One sentence about the shared approach of this cluster's papers.

Rules for "local_label":
- Use only the ids you were given, each exactly once.
- Never invent an id, and never emit -1.

Respond with JSON only."""


def _one_line(text: str) -> str:
    """Collapse whitespace and truncate, so one title stays one line."""
    collapsed = " ".join((text or "").split())
    return collapsed if len(collapsed) <= MAX_TITLE_CHARS else collapsed[: MAX_TITLE_CHARS - 1] + "…"


def build_cluster_naming_prompt(
    topic: str,
    clusters: list[dict],
    exemplar_titles: dict[int, list[str]],
) -> str:
    """One block per real cluster: id, size, and up to :data:`MAX_EXEMPLARS` titles.

    ``-1`` never reaches this function (``name_clusters`` filters it out); the
    builder does not special-case it beyond rendering whatever it is handed.
    """
    lines = [f"TOPIC: {topic}", "", "CLUSTERS:"]
    for cluster in clusters:
        local_label = int(cluster.get("local_label", -1))
        all_titles = exemplar_titles.get(local_label, [])
        size = int(cluster.get("size", cluster.get("paper_count", len(all_titles))))
        lines.append(f"cluster {local_label} - {size} paper(s):")
        titles = all_titles[:MAX_EXEMPLARS]
        if titles:
            lines.extend(f"  - {_one_line(title)}" for title in titles)
        else:
            lines.append("  - (no exemplar titles available)")
        lines.append("")
    return "\n".join(lines).rstrip()


__all__ = ["CLUSTER_LABEL_SYSTEM_PROMPT", "MAX_EXEMPLARS", "build_cluster_naming_prompt"]