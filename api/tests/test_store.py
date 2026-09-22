"""Tests for api/store.py — the plan §11.4 row for store:

migrate() upgrades a V1-shaped database with all rows intact; extraction cache
keyed by (paper_id, prompt_version); session() rolls back on exception.
Plus the A.3 surface: CRUD round-trips, citations, source_cache TTL, runs
replay, llm_calls rollup, prune, cascade delete.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
import store
from models import PaperExtraction
from store import StoreError

from conftest import make_paper
