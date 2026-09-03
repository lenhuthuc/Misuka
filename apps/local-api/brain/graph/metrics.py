"""One JSONL line per turn: what it cost, which backend paid, what the guard saw.

Separate from `logs/mitsuka.log` on purpose. That file is a narrative for a
human reading along during development; this one is a table, and the questions
it exists to answer are counting questions -- how many turns a day, what
fraction fell back to local, how often the guard fires and on what. Mixing them
means grepping prose to get a number.

The line is written even when the turn failed. A turn that errored is exactly
the kind you want in the denominator.

Analysis, with the venv python:

    # turns per day, and the local-fallback share
    import json, collections
    rows = [json.loads(l) for l in open("logs/turns.jsonl", encoding="utf-8")]
    per_day = collections.Counter(r["ts"][:10] for r in rows)
    local = sum(r["route"] == "local" for r in rows) / max(len(rows), 1)
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class TurnMetrics:
    """Append-only JSONL sink. Safe to share across concurrent turns."""

    def __init__(self, path: Path, enabled: bool = True) -> None:
        self._path = path
        self._enabled = enabled
        # Writes are a few hundred bytes and land under the same lock, so an
        # interleaved half-line can never appear. A queue and a writer task
        # would be the scalable shape; at one conversation per machine it would
        # be machinery with nothing to do.
        self._lock = threading.Lock()
        if enabled:
            path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict[str, Any]) -> None:
        """Write one turn. Never raises -- measurement must not break a turn."""
        if not self._enabled:
            return
        line = json.dumps(
            {"ts": datetime.now(timezone.utc).isoformat(), **record},
            ensure_ascii=False,
            default=str,
        )
        try:
            with self._lock, self._path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except OSError:
            logger.exception("turn metrics write failed | path=%s", self._path)


def build_record(
    *,
    session_id: str,
    turn_id: str,
    route: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    latency_ms: float,
    flags: list[str],
    guard_action: str = "pass",
    regenerated: bool = False,
    fallback_reason: str = "",
    streamed: bool = False,
    error: str = "",
) -> dict[str, Any]:
    """The turn record's shape, in one place so the columns stay stable.

    Deliberately no prompt or reply text: `core/logging.py` sets the rule for
    this app ("never log raw prompt/response text"), and a metrics file is the
    easiest place to break it by accident.
    """
    return {
        "session_id": session_id,
        "turn_id": turn_id,
        "route": route,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "latency_ms": round(latency_ms, 1),
        "flags": flags,
        "guard_action": guard_action,
        "regenerated": regenerated,
        "fallback_reason": fallback_reason,
        "streamed": streamed,
        "error": error,
    }
