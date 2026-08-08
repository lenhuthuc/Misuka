from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiosqlite

logger = logging.getLogger(__name__)

_DDL = """
CREATE TABLE IF NOT EXISTS conversations (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    role      TEXT    NOT NULL,
    content   TEXT    NOT NULL,
    timestamp TEXT    NOT NULL,
    valence   REAL,
    arousal   REAL,
    dominance REAL,
    emotion   TEXT
);

CREATE TABLE IF NOT EXISTS facts (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    key          TEXT    NOT NULL UNIQUE,
    value        TEXT    NOT NULL,
    embedding_id TEXT,
    updated_at   TEXT    NOT NULL
);

-- Exchanges waiting to be distilled into compact memories. Durable on purpose:
-- curation is deferred until the LLM runner is quiet, so anything held only in
-- memory would be lost on shutdown or whenever the user keeps talking.
CREATE TABLE IF NOT EXISTS pending_curation (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    query           TEXT    NOT NULL,
    response        TEXT    NOT NULL,
    emotion         TEXT,
    valence         REAL,
    arousal         REAL,
    dominance       REAL,
    created_at      TEXT    NOT NULL,
    attempts        INTEGER NOT NULL DEFAULT 0
);
"""


class MemoryService:
    """Async SQLite-backed structured memory (conversations + facts)."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: aiosqlite.Connection | None = None

    # Columns added after the initial release — applied via ALTER TABLE on old DBs
    _VAD_COLUMNS = (("valence", "REAL"), ("arousal", "REAL"), ("dominance", "REAL"), ("emotion", "TEXT"))

    async def initialize(self) -> None:
        self._conn = await aiosqlite.connect(str(self._db_path))
        self._conn.row_factory = aiosqlite.Row
        await self._conn.executescript(_DDL)
        await self._migrate()
        await self._conn.commit()
        logger.info("MemoryService initialized at %s", self._db_path)

    async def _migrate(self) -> None:
        assert self._conn is not None
        cur = await self._conn.execute("PRAGMA table_info(conversations)")
        existing = {row["name"] for row in await cur.fetchall()}
        for name, sql_type in self._VAD_COLUMNS:
            if name not in existing:
                await self._conn.execute(f"ALTER TABLE conversations ADD COLUMN {name} {sql_type}")
                logger.info("MemoryService migration | added conversations.%s", name)

    async def close(self) -> None:
        if self._conn:
            await self._conn.close()

    async def __aenter__(self) -> "MemoryService":
        await self.initialize()
        return self

    async def __aexit__(self, *_) -> None:
        await self.close()

    async def save_message(
        self,
        role: str,
        content: str,
        vad: tuple[float, float, float] | None = None,
        emotion: str | None = None,
    ) -> int:
        ts = datetime.now(timezone.utc).isoformat()
        v, a, d = vad if vad else (None, None, None)
        assert self._conn is not None
        cur = await self._conn.execute(
            """INSERT INTO conversations (role, content, timestamp, valence, arousal, dominance, emotion)
               VALUES (?,?,?,?,?,?,?)""",
            (role, content, ts, v, a, d, emotion),
        )
        await self._conn.commit()
        logger.debug("Saved message | role=%s len=%d emotion=%s", role, len(content), emotion)
        return cur.lastrowid  # type: ignore[return-value]

    async def get_recent(self, limit: int = 10) -> list[dict[str, Any]]:
        assert self._conn is not None
        cur = await self._conn.execute(
            """SELECT role, content, timestamp, valence, arousal, dominance, emotion
               FROM conversations ORDER BY id DESC LIMIT ?""",
            (limit,),
        )
        rows = await cur.fetchall()
        return [dict(r) for r in reversed(rows)]

    # ── Curation queue ───────────────────────────────────────────────────────

    async def enqueue_curation(
        self,
        query: str,
        response: str,
        emotion: str | None = None,
        vad: tuple[float, float, float] | None = None,
    ) -> int:
        """Record an exchange as owing fact extraction.

        Use when: a turn has just finished and its raw text is already stored,
        but the LLM work that would mine it for durable facts must wait for a
        quiet runner.

        Returns: the queue row id.
        """
        ts = datetime.now(timezone.utc).isoformat()
        v, a, d = vad if vad else (None, None, None)
        assert self._conn is not None
        cur = await self._conn.execute(
            """INSERT INTO pending_curation
               (query, response, emotion, valence, arousal, dominance, created_at)
               VALUES (?,?,?,?,?,?,?)""",
            (query, response, emotion, v, a, d, ts),
        )
        await self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    async def next_curation_batch(self, limit: int, max_attempts: int = 3) -> list[dict[str, Any]]:
        """Oldest pending exchanges, skipping ones that have failed too often.

        Rows are read rather than claimed: a single worker drains this queue, so
        there is no second consumer to race with, and leaving them in place is
        what makes an abandoned batch retryable.
        """
        assert self._conn is not None
        cur = await self._conn.execute(
            """SELECT id, query, response, emotion, valence, arousal, dominance
               FROM pending_curation WHERE attempts < ? ORDER BY id ASC LIMIT ?""",
            (max_attempts, limit),
        )
        rows = await cur.fetchall()
        return [dict(r) for r in rows]

    async def complete_curation(self, ids: list[int]) -> None:
        """Drop rows whose memories are now stored."""
        if not ids:
            return
        assert self._conn is not None
        placeholders = ",".join("?" * len(ids))
        await self._conn.execute(f"DELETE FROM pending_curation WHERE id IN ({placeholders})", ids)
        await self._conn.commit()

    async def record_curation_failure(self, ids: list[int]) -> None:
        """Count an attempt against rows so a poisonous batch cannot loop forever."""
        if not ids:
            return
        assert self._conn is not None
        placeholders = ",".join("?" * len(ids))
        await self._conn.execute(
            f"UPDATE pending_curation SET attempts = attempts + 1 WHERE id IN ({placeholders})", ids
        )
        await self._conn.commit()

    async def pending_curation_count(self, max_attempts: int = 3) -> int:
        assert self._conn is not None
        cur = await self._conn.execute(
            "SELECT COUNT(*) AS n FROM pending_curation WHERE attempts < ?", (max_attempts,)
        )
        row = await cur.fetchone()
        return row["n"] if row else 0

    async def upsert_fact(self, key: str, value: str, embedding_id: str | None = None) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        assert self._conn is not None
        await self._conn.execute(
            """INSERT INTO facts (key, value, embedding_id, updated_at) VALUES (?,?,?,?)
               ON CONFLICT(key) DO UPDATE SET value=excluded.value,
                   embedding_id=excluded.embedding_id, updated_at=excluded.updated_at""",
            (key, value, embedding_id, ts),
        )
        await self._conn.commit()

    async def get_fact(self, key: str) -> str | None:
        assert self._conn is not None
        cur = await self._conn.execute("SELECT value FROM facts WHERE key=?", (key,))
        row = await cur.fetchone()
        return row["value"] if row else None

    async def list_facts(self) -> list[dict[str, Any]]:
        """All stored facts, most recently updated first.

        Ordering is not cosmetic: facts go into every system prompt under a
        character budget, so when the table outgrows that budget the order
        decides which facts the model still sees.
        """
        assert self._conn is not None
        cur = await self._conn.execute(
            "SELECT key, value, embedding_id, updated_at FROM facts ORDER BY updated_at DESC"
        )
        rows = await cur.fetchall()
        return [dict(r) for r in rows]
