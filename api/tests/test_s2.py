"""Tests for the Semantic Scholar source — Phase 3, Task 3.2.

All driven through ``httpx.MockTransport``; nothing touches the network. The
pacing test monkeypatches ``time.monotonic`` and ``time.sleep`` so it asserts
the 1 RPS contract without actually waiting a second.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx

from config import Settings
from sources import semanticscholar as s2

FIXTURES = Path(__file__).parent / "fixtures"


def _settings(tmp_path, **overrides: Any) -> Settings:
    base = replace(
        Settings.from_env(),
        arxiv_offline=True,
        s2_api_key="",
    )
    for name, value in overrides.items():
        object.__setattr__(base, name, value)
    return base


def _transport(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), base_url=s2._BASE)


def test_fetch_paper_200(tmp_path):
    settings = _settings(tmp_path)
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json=json.loads((FIXTURES / "s2_paper.json").read_text()))

    payload = s2.fetch_paper("2005.11401", settings, client=_transport(handler))
    assert payload is not None
    assert payload["citationCount"] == 18451
    assert payload["externalIds"]["ArXiv"] == "2005.11401"
    assert calls[0].url.path == "/graph/v1/paper/arXiv:2005.11401"
    assert "citationCount" in calls[0].url.params["fields"]


def test_api_key_header_is_sent_only_when_set(tmp_path):
    keyed = _settings(tmp_path, s2_api_key="sk-test")
    headers: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        headers.append(dict(request.headers))
        return httpx.Response(200, json={})

    s2.fetch_paper("x", keyed, client=_transport(handler))
    assert headers[0].get("x-api-key") == "sk-test"

    keyless = _settings(tmp_path, s2_api_key="")
    headers.clear()
    s2.fetch_paper("x", keyless, client=_transport(handler))
    assert "x-api-key" not in headers[0]


def test_429_then_200_succeeds(tmp_path):
    settings = _settings(tmp_path)
    attempts = {"n": 0}
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "0"}, json={"error": "rate"})
        return httpx.Response(200, json={"paperId": "abc", "citationCount": 5})

    orig_sleep = s2.time.sleep
    s2.time.sleep = sleeps.append  # type: ignore[assignment]
    try:
        payload = s2.fetch_paper("x", settings, client=_transport(handler))
    finally:
        s2.time.sleep = orig_sleep
    assert payload == {"paperId": "abc", "citationCount": 5}
    assert attempts["n"] == 2
    assert sleeps  # the retry waited, honouring Retry-After


def test_persistent_429_gives_up_and_returns_none(tmp_path):
    settings = _settings(tmp_path)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"Retry-After": "0"}, json={"error": "rate"})

    orig_sleep = s2.time.sleep
    s2.time.sleep = lambda *_: None  # type: ignore[assignment]
    try:
        assert s2.fetch_paper("x", settings, client=_transport(handler)) is None
        assert s2.fetch_references("x", settings, client=_transport(handler)) == []
        assert s2.fetch_citations("x", settings, client=_transport(handler)) == []
    finally:
        s2.time.sleep = orig_sleep


def test_transport_failure_returns_none_rather_than_raising(tmp_path):
    settings = _settings(tmp_path)

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("blackholed")

    assert s2.fetch_paper("x", settings, client=_transport(handler)) is None


def test_response_is_cached_for_seven_days(tmp_path, conn):
    settings = _settings(tmp_path)
    payload = json.loads((FIXTURES / "s2_paper.json").read_text())
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=payload)

    client = _transport(handler)
    assert s2.fetch_paper("2005.11401", settings, client=client, conn=conn) == payload
    assert s2.fetch_paper("2005.11401", settings, client=client, conn=conn) == payload
    assert len(requests) == 1  # second call came from source_cache


def test_pacing_lock_prevents_two_calls_within_one_second(monkeypatch):
    """The 1 RPS contract: the second request must be spaced, not immediate."""
    timeline = {"now": 100.0}

    def fake_monotonic() -> float:
        return timeline["now"]

    def fake_sleep(seconds: float) -> None:
        timeline["now"] += seconds

    monkeypatch.setattr(s2.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(s2.time, "sleep", fake_sleep)

    s2._last_request_at = 0.0  # reset module state
    s2._pace()  # first call: no wait
    assert timeline["now"] == 100.0
    s2._pace()  # second call immediately after: must wait ~1s
    assert timeline["now"] == 101.0
    s2._pace()  # immediately after that: another full second
    assert timeline["now"] == 102.0
