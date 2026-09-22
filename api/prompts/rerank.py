"""B.2 — LLM-as-judge reranking prompt.

Keyword overlap is not relevance: a paper can match every term in a topic and
still be a passing mention. The cross-encoder handles most of this, but it is
trained on web passages and is unreliable for research prose, so a judge that
reads the topic and each abstract corrects the top of the ranking.

The judge returns scores on the same 0-10 scale the cross-encoder produces, so
blending needs no rescaling. It answers with the ``JudgeBatch`` envelope
(``{"scores": [{"paper_id": str, "relevance_0_10": number}]}``) — exactly one
entry per supplied paper id, no new ids, no prose outside JSON.
"""

from __future__ import annotations

from models import Paper

#: B.2 budgets the batch at <=6k tokens in: passages are truncated to 900
#: characters each, ten to a batch.
MAX_PASSAGE_CHARS = 900

JUDGE_SYSTEM_PROMPT = """You are a relevance judge for an academic search engine.

You are given a TOPIC and a list of papers, each tagged with its paper_id. For
each paper, score how relevant it is to the topic, from 0 to 10:

- 9-10: a paper someone entering this field must read; centrally about the topic
- 7-8: a direct, on-topic contribution
- 4-6: adjacent — a relevant method or application, but about something else
- 1-3: mentions the topic but is about something else
- 0: unrelated

Ask for every paper: does it ADVANCE this topic, or merely mention it?
A famous paper that only mentions the topic scores lower than an obscure paper
entirely about it. A survey that merely lists the topic among many others is
3-4, not 9.

Respond with JSON only, in exactly this shape:
{"scores": [{"paper_id": "<the id you were given>", "relevance_0_10": 8}]}

Include every paper_id you were given, exactly once, and no new ids."""


def build_judge_user_prompt(topic: str, papers: list[Paper]) -> str:
    """One block per paper, tagged with its paper_id — the alignment contract."""
    blocks = []
    for paper in papers:
        text = paper.rerank_text[:MAX_PASSAGE_CHARS]
        blocks.append(f"[{paper.paper_id}] {text}")
    joined = "\n\n".join(blocks)
    return f"TOPIC: {topic}\n\nPAPERS:\n\n{joined}"
