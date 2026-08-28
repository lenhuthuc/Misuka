from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiosqlite

from brain.bm25_repetition import BM25RepetitionIndex

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

    def __init__(
        self,
        db_path: Path,
        *,
        bm25_repetition_threshold: float = 0.78,
        bm25_repetition_min_tokens: int = 5,
        bm25_repetition_max_sentences: int = 2000,
    ) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: aiosqlite.Connection | None = None
        self._bm25 = BM25RepetitionIndex(
            db_path.parent / "bm25_index.json",
            threshold=bm25_repetition_threshold,
            min_tokens=bm25_repetition_min_tokens,
            max_sentences=bm25_repetition_max_sentences,
        )

    # Columns added after the initial release — applied via ALTER TABLE on old DBs
    _VAD_COLUMNS = (("valence", "REAL"), ("arousal", "REAL"), ("dominance", "REAL"), ("emotion", "TEXT"))

    async def initialize(self) -> None:
        self._conn = await aiosqlite.connect(str(self._db_path))
        self._conn.row_factory = aiosqlite.Row
        await self._conn.executescript(_DDL)
        await self._migrate()
        await self._conn.commit()
        if not self._bm25.load():
            cur = await self._conn.execute(
                "SELECT id, content FROM conversations WHERE role = 'assistant' ORDER BY id"
            )
            self._bm25.rebuild((row["id"], row["content"]) for row in await cur.fetchall())
            logger.info("BM25 sentence index rebuilt | sentences=%d", self._bm25.size)
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
        message_id = cur.lastrowid
        if role == "assistant" and message_id is not None:
            try:
                self._bm25.add_message(message_id, content)
            except OSError:
                logger.exception("Failed to persist BM25 sentence index")
        logger.debug("Saved message | role=%s len=%d emotion=%s", role, len(content), emotion)
        return message_id  # type: ignore[return-value]

    async def get_recent(self, limit: int = 10) -> list[dict[str, Any]]:
        assert self._conn is not None
        cur = await self._conn.execute(
            """SELECT id, role, content, timestamp, valence, arousal, dominance, emotion
               FROM conversations ORDER BY id DESC LIMIT ?""",
            (limit,),
        )
        rows = await cur.fetchall()
        return [dict(r) for r in reversed(rows)]

    def filter_repetitive_history(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Remove assistant sentences duplicated elsewhere in the persistent corpus."""
        filtered_rows: list[dict[str, Any]] = []
        for row in rows:
            if row.get("role") != "assistant":
                filtered_rows.append(row)
                continue
            content, removed = self._bm25.filter_text(
                str(row.get("content", "")),
                exclude_message_id=int(row["id"]),
            )
            if removed:
                logger.info("BM25 history filter | message_id=%s removed=%d", row["id"], removed)
            if content:
                filtered_rows.append({**row, "content": content})
        return filtered_rows

    def filter_assistant_response(self, text: str, *, fallback: str = "") -> str:
        filtered, removed = self._bm25.filter_text(text, fallback=fallback)
        if removed:
            logger.info("BM25 response filter | removed=%d", removed)
        return filtered
