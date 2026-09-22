"""api.llm — the LLM seam (plan Appendix A.4)."""

from llm.client import (
    LLMClient,
    LLMError,
    NullCompleter,
    build_completer,
    candidate_json_payloads,
    parse_into,
)
from llm.protocol import JSONCompleter

__all__ = [
    "JSONCompleter",
    "LLMClient",
    "LLMError",
    "NullCompleter",
    "build_completer",
    "candidate_json_payloads",
    "parse_into",
]
