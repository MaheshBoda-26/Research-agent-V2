"""Structured logging: one JSON object per line on stdout (plan §13.1).

Why JSON lines rather than prose: the stage events, the ``llm.call`` audit rows
and the per-stage timings all need to be machine-readable when a run is being
debugged, and a human can still read them. Secrets never enter a record —
``Settings.describe()`` redacts them upstream, and this module additionally
scrubs any value whose key looks like a credential.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

_SECRET_HINTS = ("key", "token", "secret", "password", "authorization")


def _scrub(value: Any, key: str = "") -> Any:
    lowered = key.lower()
    if key and any(hint in lowered for hint in _SECRET_HINTS):
        return "<redacted>" if value else ""
    if isinstance(value, dict):
        return {k: _scrub(v, k) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_scrub(v) for v in value]
    if isinstance(value, str) and len(value) > 2000:
        return value[:2000] + "…"
    return value


class JsonLineFormatter(logging.Formatter):
    """Render each record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "level": record.levelname.lower(),
            "logger": record.name,
            "event": getattr(record, "event", record.getMessage()),
        }
        if record.exc_info:
            payload["error"] = self.formatException(record.exc_info)
        extra = getattr(record, "fields", None)
        if isinstance(extra, dict):
            payload.update(_scrub(extra))
        return json.dumps(_scrub(payload), default=str, ensure_ascii=False)


def configure_logging(level: str = "INFO") -> None:
    """Install the JSON formatter on the root logger, once."""
    root = logging.getLogger()
    root.setLevel(level.upper())
    for handler in list(root.handlers):
        root.removeHandler(handler)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonLineFormatter())
    root.addHandler(handler)


def log_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    """Log one structured event (``engine.start``, ``stage.done``, …)."""
    logger.info(event, extra={"event": event, "fields": fields})


__all__ = ["JsonLineFormatter", "configure_logging", "log_event"]
