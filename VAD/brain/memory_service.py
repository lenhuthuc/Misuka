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
    timestamp TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS facts (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    key          TEXT    NOT NULL UNIQUE,
    value        TEXT    NOT NULL,
    embedding_id TEXT,
    updated_at   TEXT    NOT NULL
);
"""


class MemoryService:
    """Async SQLite-backed structured memory (conversations + facts)."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        self._conn = await aiosqlite.connect(str(self._db_path))
        self._conn.row_factory = aiosqlite.Row
        await self._conn.executescript(_DDL)
        await self._conn.commit()
        logger.info("MemoryService initialized at %s", self._db_path)

    async def close(self) -> None:
        if self._conn:
            await self._conn.close()

    async def __aenter__(self) -> "MemoryService":
        await self.initialize()
        return self

    async def __aexit__(self, *_) -> None:
        await self.close()

    async def save_message(self, role: str, content: str) -> int:
        ts = datetime.now(timezone.utc).isoformat()
        assert self._conn is not None
        cur = await self._conn.execute(
            "INSERT INTO conversations (role, content, timestamp) VALUES (?,?,?)",
            (role, content, ts),
        )
        await self._conn.commit()
        logger.debug("Saved message | role=%s len=%d", role, len(content))
        return cur.lastrowid  # type: ignore[return-value]

    async def get_recent(self, limit: int = 10) -> list[dict[str, Any]]:
        assert self._conn is not None
        cur = await self._conn.execute(
            "SELECT role, content, timestamp FROM conversations ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        rows = await cur.fetchall()
        return [dict(r) for r in reversed(rows)]

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
        assert self._conn is not None
        cur = await self._conn.execute("SELECT key, value, embedding_id, updated_at FROM facts")
        rows = await cur.fetchall()
        return [dict(r) for r in rows]
