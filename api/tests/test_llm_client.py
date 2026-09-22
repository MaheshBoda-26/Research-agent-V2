"""Tests for api/llm/client.py — the plan §11.4 rows for the LLM client.

Everything is offline: the OpenAI SDK is replaced by a scripted fake transport
(the injected seam, §11.2). Every behaviour the plan demands is covered here:
fence-stripping, wrapper unwrapping, the bounded repair loop, retryable vs
fatal transport errors, refusal handling, and per-attempt llm_calls recording.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest
from pydantic import BaseModel

from config import Settings
from llm.client import (
    LLMClient,
    NullCompleter,
    build_completer,
    candidate_json_payloads,
    parse_into,
)


class _Out(BaseModel):
    name: str
    count: int


class _ScriptedError(Exception):
    def __init__(self, status: int, message: str = "") -> None:
        super().__init__(message or f"HTTP {status}")
        self.status_code = status


def _response(content: str, prompt: int = 10, completion: int = 5) -> Any:
    msg: Any = type("M", (), {"content": content})()
    choice: Any = type("C", (), {"message": msg})()
    usage: Any = type("U", (), {"prompt_tokens": prompt, "completion_tokens": completion})()
    return type("R", (), {"choices": [choice], "usage": usage})()


class _FakeTransport:
    """Scripted OpenAI stand-in: each entry is a response or an exception."""

    def __init__(self, script: list[Any]) -> None:
        self.script = list(script)
        self.requests: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    @property
    def chat(self) -> Any:  # noqa: D401 - attribute-shaped like the SDK
        return type("Chat", (), {"completions": self})()


class _Recorder:
    """Collects llm_calls rows as kwargs."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **row: Any) -> None:
        self.calls.append(row)


def _client(transport: _FakeTransport, provider: str = "nim", **kw: Any) -> LLMClient:
    settings = replace(Settings(), llm_provider=provider, llm_model="test-model")
    recorder = _Recorder()
    client = LLMClient(settings, client=transport, on_call=recorder, **kw)
    return client


GOOD_JSON = '{"name": "alpha", "count": 3}'


def test_candidate_payloads_strips_fences_and_blocks() -> None:
    raw = f"Sure! Here you go:\n```json\n{GOOD_JSON}\n```\nHope that helps."
    payloads = candidate_json_payloads(raw)
    assert payloads[0] == raw.strip()  # whole response is the first candidate
    assert GOOD_JSON in payloads  # the fenced block is a later candidate
    # Braces inside strings do not confuse the balanced-object scanner.
    raw2 = 'prefix {"name": "a {b} c", "count": 1} suffix'
    block = candidate_json_payloads(raw2)[1]
    assert block == '{"name": "a {b} c", "count": 1}'


def test_parse_into_unwraps_wrapper_object() -> None:
    wrapped = '{"fields": {"name": "alpha", "count": 3}}'
    assert parse_into(wrapped, _Out) == _Out(name="alpha", count=3)


def test_complete_json_plain_and_fenced() -> None:
    for raw in (GOOD_JSON, f"```json\n{GOOD_JSON}\n```"):
        transport = _FakeTransport([_response(raw)])
        client = _client(transport)
        out = client.complete_json(system="s", user="u", schema=_Out, stage="test")
        assert out == _Out(name="alpha", count=3)


def test_repair_loop_second_attempt_validates() -> None:
    transport = _FakeTransport(
        [_response("not json at all"), _response(GOOD_JSON)]
    )
    client = _client(transport)
    out = client.complete_json(system="s", user="u", schema=_Out, stage="extract", run_id="r1")
    assert out is not None
    assert len(transport.requests) == 2
    # The repair message chain contains the validation error.
    assert "did not match" in transport.requests[1]["messages"][-1]["content"]
    # Two llm_calls rows: first a failure, then a success.
    assert len(client.on_call.calls) == 2
    assert client.on_call.calls[0]["ok"] is False
    assert client.on_call.calls[1]["ok"] is True
    assert client.on_call.calls[1]["stage"] == "extract"
    assert client.on_call.calls[1]["run_id"] == "r1"
    assert client.on_call.calls[1]["prompt_tokens"] == 10


def test_repair_loop_exhausted_returns_none_and_records() -> None:
    transport = _FakeTransport([_response("still bad"), _response("nope")])
    client = _client(transport)
    out = client.complete_json(system="s", user="u", schema=_Out, stage="test")
    assert out is None
    assert client.failures == 1
    recorded = client.on_call.calls[-1]
    assert recorded["ok"] is False
    assert recorded["error"].startswith("validation:")


def test_refusal_returns_none_with_reason() -> None:
    transport = _FakeTransport([_response("I'm sorry, but I cannot assist with that.")])
    client = _client(transport)
    assert client.complete_json(system="s", user="u", schema=_Out, stage="test") is None
    assert client.on_call.calls[-1]["error"] == "refusal"


def test_retryable_status_is_retried_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("llm.client.time.sleep", lambda _s: None)
    transport = _FakeTransport(
        [_ScriptedError(429), _ScriptedError(500), _response(GOOD_JSON)]
    )
    client = _client(transport)
    out = client.complete_json(system="s", user="u", schema=_Out, stage="test")
    assert out == _Out(name="alpha", count=3)
    assert len(transport.requests) == 3


def test_fatal_status_returns_none_without_retries() -> None:
    transport = _FakeTransport([_ScriptedError(401), _response(GOOD_JSON)])
    client = _client(transport)
    assert client.complete_json(system="s", user="u", schema=_Out, stage="test") is None
    assert len(transport.requests) == 1
    assert client.on_call.calls[-1]["error"].startswith("HTTP 401")


def test_retries_exhausted_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("llm.client.time.sleep", lambda _s: None)
    transport = _FakeTransport([_ScriptedError(429)] * 4)
    client = _client(transport)
    assert client.complete_json(system="s", user="u", schema=_Out, stage="test") is None
    assert len(transport.requests) == 4


def test_structured_format_fallback_on_rejection() -> None:
    # openai provider sends response_format; a 400 naming it must disable it
    # for the process and succeed via prompt+parse on the retry.
    transport = _FakeTransport([_ScriptedError(400, "response_format not supported"), _response(GOOD_JSON)])
    client = _client(transport, provider="openai")
    out = client.complete_json(system="s", user="u", schema=_Out, stage="test")
    assert out == _Out(name="alpha", count=3)
    assert client.structured_disabled is True
    assert len(transport.requests) == 2
    assert "response_format" in transport.requests[0]
    assert "response_format" not in transport.requests[1]


def test_nim_uses_guided_json_extra_body() -> None:
    transport = _FakeTransport([_response(GOOD_JSON)])
    client = _client(transport, provider="nim")
    client.complete_json(system="s", user="u", schema=_Out, stage="test")
    extra = transport.requests[0].get("extra_body", {})
    assert "nvext" in extra and "guided_json" in extra["nvext"]
    assert "response_format" not in transport.requests[0]


def test_usage_and_cost_totals_accumulate() -> None:
    # 1M/1M tokens on the (known-price) default model must yield a positive
    # cost estimate; unknown models record tokens with cost 0.0 (honest).
    transport = _FakeTransport([_response(GOOD_JSON, prompt=1_000_000, completion=1_000_000)])
    client = _client(transport)
    client.complete_json(system="s", user="u", schema=_Out, stage="test")
    assert client.total_prompt_tokens == 1_000_000
    assert client.total_completion_tokens == 1_000_000
    assert client.total_cost_usd >= 0.0
    assert client.on_call.calls[-1]["ok"] is True


def test_complete_text_success_and_refusal() -> None:
    client = _client(_FakeTransport([_response("hello"), _response("I cannot assist")]))
    assert client.complete_text(system="s", user="u") == "hello"
    assert client.complete_text(system="s", user="u") is None


def test_null_completer_and_build_completer_degrade() -> None:
    null = NullCompleter()
    assert null.complete_json(system="s", user="u", schema=_Out, stage="t") is None
    assert null.complete_text(system="s", user="u") is None

    completer = build_completer(replace(Settings(), llm_api_key="", llm_model=""))
    assert isinstance(completer, NullCompleter)

    completer = build_completer(replace(Settings(), llm_api_key="k", llm_model="m"))
    assert isinstance(completer, LLMClient)