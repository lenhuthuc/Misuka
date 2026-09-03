from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiosqlite

from brain.bm25_repetition import BM25RepetitionIndex

logger = logging.getLogger(__name__)

DEFAULT_SESSION = "default"

_DDL = """
CREATE TABLE IF NOT EXISTS conversations (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    role       TEXT    NOT NULL,
    content    TEXT    NOT NULL,
    timestamp  TEXT    NOT NULL,
    valence    REAL,
    arousal    REAL,
    dominance  REAL,
    emotion    TEXT,
    session_id TEXT    NOT NULL DEFAULT 'default'
);

-- The index on (session_id, id) is NOT here: on a database created before
-- sessions existed, `CREATE TABLE IF NOT EXISTS` is a no-op and the column
-- arrives later from `_migrate`, so an index named here would fail the whole
-- script with "no such column: session_id" before the migration could run.

-- The graph's rolling summary: what fell out of the history window, per
-- session. Deliberately alongside the transcript rather than in a second store
-- -- a checkpointer of its own would put the same conversation in two files and
-- leave RAG and the BM25 index reading the older one.
-- `summarized_through_id` is the watermark: the highest conversations.id already
-- folded into `rolling_summary`. Without it the summariser cannot tell which
-- messages have fallen out of the history window since last time -- the window
-- query only returns what is still *in* it -- and would either re-summarise the
-- same span every turn or silently drop the messages in between.
CREATE TABLE IF NOT EXISTS session_state (
    session_id           TEXT PRIMARY KEY,
    rolling_summary      TEXT    NOT NULL DEFAULT '',
    summarized_through_id INTEGER NOT NULL DEFAULT 0,
    updated_at           TEXT    NOT NULL
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

    # Columns added after the initial release — applied via ALTER TABLE on old DBs.
    # `CREATE TABLE IF NOT EXISTS` is a no-op on a database that already has the
    # table, so a new column in `_DDL` reaches existing installs only from here.
    _ADDED_COLUMNS = (
        ("valence", "REAL"),
        ("arousal", "REAL"),
        ("dominance", "REAL"),
        ("emotion", "TEXT"),
        # Every row written before sessions existed belongs to the one
        # conversation that was running, which is what `default` names.
        ("session_id", "TEXT NOT NULL DEFAULT 'default'"),
    )

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
        for name, sql_type in self._ADDED_COLUMNS:
            if name not in existing:
                await self._conn.execute(f"ALTER TABLE conversations ADD COLUMN {name} {sql_type}")
                logger.info("MemoryService migration | added conversations.%s", name)
        # Safe to run every start, and only correct once the column above is
        # guaranteed to exist -- which is why it lives here and not in `_DDL`.
        await self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_conversations_session "
            "ON conversations (session_id, id)"
        )

        cur = await self._conn.execute("PRAGMA table_info(session_state)")
        state_columns = {row["name"] for row in await cur.fetchall()}
        if state_columns and "summarized_through_id" not in state_columns:
            await self._conn.execute(
                "ALTER TABLE session_state ADD COLUMN "
                "summarized_through_id INTEGER NOT NULL DEFAULT 0"
            )
            logger.info(
                "MemoryService migration | added session_state.summarized_through_id"
            )

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
        session_id: str = DEFAULT_SESSION,
    ) -> int:
        ts = datetime.now(timezone.utc).isoformat()
        v, a, d = vad if vad else (None, None, None)
        assert self._conn is not None
        cur = await self._conn.execute(
            """INSERT INTO conversations
                   (role, content, timestamp, valence, arousal, dominance, emotion, session_id)
               VALUES (?,?,?,?,?,?,?,?)""",
            (role, content, ts, v, a, d, emotion, session_id),
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

    async def get_recent(
        self, limit: int = 10, session_id: str | None = DEFAULT_SESSION
    ) -> list[dict[str, Any]]:
        """The last `limit` messages, newest last.

        `session_id=None` reads across every session. That is not the chat
        path's business -- it exists for the maintenance scripts in `scripts/`,
        which predate sessions and look at the transcript as a whole.
        """
        assert self._conn is not None
        columns = (
            "id, role, content, timestamp, valence, arousal, dominance, emotion, session_id"
        )
        if session_id is None:
            cur = await self._conn.execute(
                f"SELECT {columns} FROM conversations ORDER BY id DESC LIMIT ?",
                (limit,),
            )
        else:
            cur = await self._conn.execute(
                f"SELECT {columns} FROM conversations WHERE session_id = ? "
                "ORDER BY id DESC LIMIT ?",
                (session_id, limit),
            )
        rows = await cur.fetchall()
        return [dict(r) for r in reversed(rows)]

    async def count_messages(self, session_id: str = DEFAULT_SESSION) -> int:
        assert self._conn is not None
        cur = await self._conn.execute(
            "SELECT COUNT(*) AS n FROM conversations WHERE session_id = ?",
            (session_id,),
        )
        row = await cur.fetchone()
        return int(row["n"]) if row else 0

    # ── Rolling summary ─────────────────────────────────────────────────────
    async def get_rolling_summary(self, session_id: str = DEFAULT_SESSION) -> str:
        summary, _ = await self.get_session_state(session_id)
        return summary

    async def get_session_state(
        self, session_id: str = DEFAULT_SESSION
    ) -> tuple[str, int]:
        """Return `(rolling_summary, summarized_through_id)` for the session."""
        assert self._conn is not None
        cur = await self._conn.execute(
            "SELECT rolling_summary, summarized_through_id FROM session_state "
            "WHERE session_id = ?",
            (session_id,),
        )
        row = await cur.fetchone()
        if row is None:
            return "", 0
        return str(row["rolling_summary"]), int(row["summarized_through_id"])

    async def set_rolling_summary(
        self,
        summary: str,
        session_id: str = DEFAULT_SESSION,
        summarized_through_id: int | None = None,
    ) -> None:
        assert self._conn is not None
        if summarized_through_id is None:
            _, summarized_through_id = await self.get_session_state(session_id)
        await self._conn.execute(
            """INSERT INTO session_state
                   (session_id, rolling_summary, summarized_through_id, updated_at)
               VALUES (?,?,?,?)
               ON CONFLICT(session_id) DO UPDATE
                   SET rolling_summary       = excluded.rolling_summary,
                       summarized_through_id = excluded.summarized_through_id,
                       updated_at            = excluded.updated_at""",
            (
                session_id,
                summary,
                summarized_through_id,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        await self._conn.commit()
        logger.debug(
            "Rolling summary saved | session=%s len=%d through_id=%d",
            session_id, len(summary), summarized_through_id,
        )

    async def get_messages_between(
        self,
        after_id: int,
        before_id: int,
        session_id: str = DEFAULT_SESSION,
    ) -> list[dict[str, Any]]:
        """Messages that have fallen out of the history window, oldest first.

        The half-open span `(after_id, before_id)` is exactly "newer than what
        the summary already covers, older than what the prompt still carries
        verbatim" — the messages that would otherwise be forgotten.
        """
        assert self._conn is not None
        if before_id <= after_id + 1:
            return []
        cur = await self._conn.execute(
            """SELECT id, role, content, timestamp FROM conversations
               WHERE session_id = ? AND id > ? AND id < ?
               ORDER BY id""",
            (session_id, after_id, before_id),
        )
        return [dict(row) for row in await cur.fetchall()]

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
