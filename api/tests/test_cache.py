"""Cache tests — Phase 2, Task 2.3.

Fresh hit inside the TTL, miss after it, expired-but-``allow_stale`` hit, and
the offline + cold-cache ``RetrievalOffline`` path. All filesystem work is
isolated to ``tmp_path``; freshness is decided by mtime, so the tests age files
by backdating it directly.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import replace
from pathlib import Path

import pytest
from conftest import make_paper

from config import Settings
from pipeline.retrieve import (
    RetrievalOffline,
    RetrievalThrottled,
    cache_key,
    fetch_candidates,
    read_cache,
    write_cache,
)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    base = replace(
        Settings.from_env(),
        db_path=tmp_path / "test.db",
        retrieval_cache_dir=tmp_path / "cache",
        retrieval_max_results=5,
        arxiv_offline=False,
    )
    return base


def _age_cache_file(settings: Settings, key: str, *, hours: float) -> None:
    """Backdate a cache file's mtime — the field freshness is decided by."""
    path = settings.retrieval_cache_dir / f"{key}.json"
    old = time.time() - hours * 3600
    os.utime(path, (old, old))


def test_cache_key_is_stable_and_input_sensitive(settings: Settings):
    assert cache_key("a b", 5) == cache_key("a b", 5)
    assert cache_key("a b", 5) != cache_key("a  b", 5)
    assert cache_key("a b", 5) != cache_key("a b", 6)


def test_write_then_read_round_trips(settings: Settings):
    papers = [make_paper(paper_id="x"), make_paper(paper_id="y")]
    key = cache_key("topic", 5)
    write_cache(settings, key, "all:topic", papers)
    restored = read_cache(settings, key)
    assert restored == papers


def test_expired_cache_is_a_miss(settings: Settings):
    key = cache_key("topic", 5)
    write_cache(settings, key, "q", [make_paper()])
    _age_cache_file(settings, key, hours=25)
    assert read_cache(settings, key) is None


def test_expired_cache_is_served_when_stale_is_allowed(settings: Settings):
    key = cache_key("topic", 5)
    write_cache(settings, key, "q", [make_paper()])
    _age_cache_file(settings, key, hours=999)
    assert read_cache(settings, key, allow_stale=True) is not None


def test_ttl_zero_disables_expiry(settings: Settings):
    no_expiry = replace(settings, retrieval_cache_ttl_hours=0)
    key = cache_key("topic", 5)
    write_cache(no_expiry, key, "q", [make_paper()])
    _age_cache_file(no_expiry, key, hours=999)
    assert read_cache(no_expiry, key) is not None


def test_corrupt_cache_is_ignored_rather_than_raising(settings: Settings):
    key = cache_key("topic", 5)
    settings.retrieval_cache_dir.mkdir(parents=True, exist_ok=True)
    (settings.retrieval_cache_dir / f"{key}.json").write_text("{not json")
    assert read_cache(settings, key) is None
    assert read_cache(settings, key, allow_stale=True) is None


def test_cache_skips_individual_malformed_papers(settings: Settings):
    key = cache_key("topic", 5)
    settings.retrieval_cache_dir.mkdir(parents=True, exist_ok=True)
    (settings.retrieval_cache_dir / f"{key}.json").write_text(
        json.dumps({"papers": [make_paper().model_dump(), {"paper_id": "broken"}]})
    )
    restored = read_cache(settings, key)
    assert restored is not None and len(restored) == 1


def test_offline_without_cache_names_the_topic(settings: Settings):
    offline = replace(settings, arxiv_offline=True)
    with pytest.raises(RetrievalOffline) as excinfo:
        fetch_candidates("quantum error correction for ML", offline)
    assert "quantum error correction for ML" in str(excinfo.value)


def test_offline_serves_a_warm_cache(settings: Settings):
    offline = replace(settings, arxiv_offline=True)
    topic = "sparse attention"
    key = cache_key(topic, 5)
    write_cache(offline, key, "q", [make_paper(paper_id="x")])
    papers = fetch_candidates(topic, offline)
    assert [p.paper_id for p in papers] == ["x"]


def test_offline_serves_an_expired_cache_deliberately(settings: Settings):
    """Offline mode is the 'arXiv unreachable' case, so stale data wins."""
    offline = replace(settings, arxiv_offline=True)
    topic = "sparse attention"
    key = cache_key(topic, 5)
    write_cache(offline, key, "q", [make_paper(paper_id="stale")])
    _age_cache_file(offline, key, hours=99)
    papers = fetch_candidates(topic, offline)
    assert [p.paper_id for p in papers] == ["stale"]


def test_force_429_raises_throttled_when_nothing_is_cached(settings: Settings):
    throttled = replace(settings, arxiv_offline=False, arxiv_force_429=True)
    with pytest.raises(RetrievalThrottled) as excinfo:
        fetch_candidates("a topic with no cache", throttled)
    assert "429" in str(excinfo.value) or "throttl" in str(excinfo.value)


def test_force_429_falls_back_to_stale_cache(settings: Settings):
    """Throttling is the common case; stale results beat a failed run."""
    topic = "throttled topic"
    key = cache_key(topic, 5)
    write_cache(settings, key, "q", [make_paper(paper_id="stale")])
    _age_cache_file(settings, key, hours=99)

    throttled = replace(settings, arxiv_offline=False, arxiv_force_429=True)
    papers = fetch_candidates(topic, throttled)
    assert [p.paper_id for p in papers] == ["stale"]
