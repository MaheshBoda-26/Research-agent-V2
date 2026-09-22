"""Tests for the source_cache TTL store — Phase 3, Task 3.1."""

from __future__ import annotations

import json
import time

from config import Settings
from store import get_cached, put_cached


def test_put_then_get_round_trips(conn):
    payload = {"citationCount": 18451}
    put_cached(conn, "s2:paper:arXiv:2005.11401", "semanticscholar", payload, ttl_seconds=3600)
    assert get_cached(conn, "s2:paper:arXiv:2005.11401", "semanticscholar") == payload


def test_get_is_scoped_by_source(conn):
    put_cached(conn, "key", "semanticscholar", {"a": 1}, ttl_seconds=3600)
    assert get_cached(conn, "key", "openalex") is None
    assert get_cached(conn, "key", "semanticscholar") == {"a": 1}


def test_expired_entry_is_a_miss(conn, settings: Settings):
    # A TTL of 0 makes the entry expire immediately (same-second timestamps).
    put_cached(conn, "key", "s", {"a": 1}, ttl_seconds=0)
    time.sleep(1.1)  # ISO strings have second precision; cross the boundary
    assert get_cached(conn, "key", "s") is None


def test_corrupt_payload_returns_none_rather_than_raising(conn):
    conn.execute(
        "INSERT INTO source_cache (key, source, payload_json, fetched_at, expires_at) "
        "VALUES ('k', 's', '{not json', '2026-01-01T00:00:00Z', '2099-01-01T00:00:00Z')"
    )
    assert get_cached(conn, "k", "s") is None


def test_expired_row_compared_as_iso_strings(conn):
    """Expiry uses the ISO string compare; a far-future expiry is a hit."""
    payload = json.dumps({"v": 1})
    conn.execute(
        "INSERT INTO source_cache (key, source, payload_json, fetched_at, expires_at) "
        "VALUES ('k3', 's', ?, '2026-01-01T00:00:00Z', '2099-01-01T00:00:00Z')",
        (payload,),
    )
    assert get_cached(conn, "k3", "s") == {"v": 1}


def test_put_overwrites_the_previous_payload_and_expiry(conn):
    put_cached(conn, "k", "s", {"v": 1}, ttl_seconds=0)
    put_cached(conn, "k", "s", {"v": 2}, ttl_seconds=3600)
    assert get_cached(conn, "k", "s") == {"v": 2}
