"""Shared test fixtures.

Every fixture is hermetic: the database, the arXiv response cache, and the
model cache all live under ``tmp_path``. No test in this suite may touch the
network or a model — everything external is an injected seam (plan §11.2).

Paths are injected by ``dataclasses.replace`` on a frozen ``Settings``, which
is why ``Settings`` is a frozen dataclass rather than a mutable object.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest

API_ROOT = Path(__file__).resolve().parent.parent
if str(API_ROOT) not in sys.path:  # pragma: no cover - defensive; pyproject pythonpath covers it
    sys.path.insert(0, str(API_ROOT))

# Tests must never reach the network. The arXiv API rate-limits aggressively,
# so a suite that depends on it fails for reasons unrelated to the code.
os.environ.setdefault("ARXIV_OFFLINE", "1")

from config import Settings  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Isolated settings rooted at tmp_path, with a freshly created schema."""
    base = Settings.from_env()
    isolated = replace(
        base,
        llm_api_key="",
        llm_model="",
        db_path=tmp_path / "landscapes.db",
        retrieval_cache_dir=tmp_path / "cache" / "arxiv",
        arxiv_offline=True,
    )
    isolated.ensure_dirs()
    import store

    store.init_db(isolated.db_path)
    return isolated


@pytest.fixture
def conn(settings: Settings) -> Iterator:
    """A live connection to the isolated database."""
    import store

    connection = store.connect(settings.db_path)
    try:
        yield connection
    finally:
        connection.close()


def make_paper(paper_id: str = "2107.05580", version: str = "1", **overrides: object):
    """A minimal valid Paper, overridable per test."""
    from models import Paper

    base: dict = {
        "paper_id": paper_id,
        "version": version,
        "title": f"Title for {paper_id}",
        "abstract": f"Abstract for {paper_id}.",
        "authors": ["A. Author"],
        "published": "2021-07-12T00:00:00Z",
        "updated": "2021-07-12T00:00:00Z",
        "primary_category": "cs.CL",
        "categories": ["cs.CL"],
        "abs_url": f"https://arxiv.org/abs/{paper_id}",
        "pdf_url": f"https://arxiv.org/pdf/{paper_id}",
    }
    base.update(overrides)
    return Paper(**base)
