from __future__ import annotations

from datetime import datetime, timezone


def utc_now_iso() -> str:
    """Return UTC timestamp like 2026-01-28T12:34:56Z."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

