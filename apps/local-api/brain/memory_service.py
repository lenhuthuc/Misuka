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
"""


class MemoryService:
    """Async SQLite-backed conversation history.

    Durable memory beyond the history window belongs to the vector store
    (`brain.vector_service`), which indexes every finished exchange. This class
    holds only the verbatim transcript the prompt's recent-history block is
    built from.
    """

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
