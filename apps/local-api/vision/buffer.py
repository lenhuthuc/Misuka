"""In-RAM store for what was learned about the images of a session.

Nothing here touches disk. The original upload is discarded the moment ingest
finishes; what is retained is the *derived text* (detections, OCR lines, the
fast summary) plus a small JPEG thumbnail that exists only so a later VLM call
has something to send. The thumbnail is the only part with an expiry -- the
text stays for the life of the session, because it is what the fast path
answers from and it is already the same order of size as a chat message.

`persist()` is a deliberate no-op seam: the buffer is a dict today, and the
hook marks where a durable store would attach without any caller changing.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ImageRecord:
    """Everything the pipeline knows about one ingested image."""

    image_id: str
    session_id: str
    ts: float
    tags: list[str] = field(default_factory=list)
    # [{label, score, box:[x, y, w, h]}], already filtered by det_score_min.
    detections: list[dict] = field(default_factory=list)
    # [{text, score, box:[x, y, w, h]}], filtered by ocr_score_min, reading order.
    ocr: list[dict] = field(default_factory=list)
    # [{key, value}] inferred from OCR layout -- see summary.build_ocr_pairs.
    ocr_pairs: list[dict] = field(default_factory=list)
    clip_emb: np.ndarray | None = None
    fast_summary: str = ""
    # JPEG bytes, dropped once `thumb_expires_at` passes.
    thumb: bytes | None = None
    thumb_expires_at: float = 0.0
    width: int = 0
    height: int = 0
    vlm_caption: str | None = None
    # [{q, a, q_emb}] -- answers already paid for, keyed by question embedding.
    vlm_qa: list[dict] = field(default_factory=list)

    def thumb_alive(self, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        return self.thumb is not None and now < self.thumb_expires_at

    def expire_thumb(self) -> None:
        self.thumb = None


class ImageBuffer:
    """Per-session ring of `ImageRecord`, oldest evicted first.

    Guarded by a plain `threading.Lock` rather than an asyncio one: ingest runs
    its models in a thread pool, so records are touched from worker threads as
    well as the event loop, and every critical section here is a few dict ops.
    """

    def __init__(self, max_per_session: int = 8) -> None:
        self._max = max_per_session
        self._sessions: dict[str, list[ImageRecord]] = {}
        self._counters: dict[str, int] = {}
        self._lock = threading.Lock()

    # -- writes --------------------------------------------------------------

    def next_image_id(self, session_id: str) -> str:
        with self._lock:
            n = self._counters.get(session_id, 0) + 1
            self._counters[session_id] = n
        return f"img_{n}"

    def add(self, record: ImageRecord) -> None:
        with self._lock:
            records = self._sessions.setdefault(record.session_id, [])
            records.append(record)
            while len(records) > self._max:
                dropped = records.pop(0)
                dropped.expire_thumb()
                logger.debug(
                    "vision.buffer evicted %s from session (cap %d)",
                    dropped.image_id, self._max,
                )
        self.persist(record)

    # -- reads ---------------------------------------------------------------

    def list_session(self, session_id: str, *, now: float | None = None) -> list[ImageRecord]:
        """Oldest-first records for a session, thumbnails swept first."""
        self.sweep(now=now)
        with self._lock:
            return list(self._sessions.get(session_id, ()))

    def latest(self, session_id: str, *, now: float | None = None) -> ImageRecord | None:
        records = self.list_session(session_id, now=now)
        return records[-1] if records else None

    def get(self, session_id: str, image_id: str) -> ImageRecord | None:
        with self._lock:
            for record in self._sessions.get(session_id, ()):
                if record.image_id == image_id:
                    return record
        return None

    # -- maintenance ---------------------------------------------------------

    def sweep(self, *, now: float | None = None) -> int:
        """Drop thumbnails past their TTL. Returns how many were freed."""
        now = time.time() if now is None else now
        freed = 0
        with self._lock:
            for records in self._sessions.values():
                for record in records:
                    if record.thumb is not None and now >= record.thumb_expires_at:
                        record.expire_thumb()
                        freed += 1
        if freed:
            logger.info("vision.buffer expired %d thumbnail(s)", freed)
        return freed

    def clear_session(self, session_id: str) -> None:
        with self._lock:
            for record in self._sessions.pop(session_id, ()):
                record.expire_thumb()
            self._counters.pop(session_id, None)

    def clear(self) -> None:
        with self._lock:
            self._sessions.clear()
            self._counters.clear()

    def persist(self, record: ImageRecord) -> None:
        """Hook for a durable store. Intentionally empty -- see module docstring."""
