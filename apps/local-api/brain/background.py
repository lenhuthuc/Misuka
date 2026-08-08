"""Post-response work for a chat turn: persist it, index it, queue it for curation.

Everything here runs after the HTTP response has been returned, and — by
design — none of it calls the LLM. The distillation that used to happen inline
now belongs to `brain.curator`, because on this CPU the runner is serialised:
an LLM call issued here lands in front of the user's next turn. Measured over a
five-turn conversation, every inline extraction attempt was either blocking or
abandoned, so nothing was ever extracted. Enqueuing instead makes the work
durable and lets the curator batch it.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from brain.curator import MemoryCurator
    from brain.emotion_service import EmotionReading
    from brain.memory_service import MemoryService
    from brain.vector_service import VectorService

logger = logging.getLogger(__name__)


async def run_memory_tasks(
    query: str,
    response: str,
    memory: "MemoryService",
    emotion: "EmotionReading | None" = None,
    vector: "VectorService | None" = None,
    curator: "MemoryCurator | None" = None,
) -> None:
    """Save the turn, index it for retrieval, and queue it for fact extraction.

    Use when: a chat turn has finished and its response has been sent.

    Expects: to be spawned as a background task, never awaited by a request.

    Indexing here costs only an embedding, so it happens immediately and the
    exchange is retrievable straight away. The curator's work is queued rather
    than done here because it needs the LLM runner, which belongs to whoever is
    waiting on a reply.
    """
    try:
        await memory.save_message("user", query)
        await memory.save_message(
            "assistant", response,
            vad=emotion.vad if emotion else None,
            emotion=emotion.emotion if emotion else None,
        )
        logger.info("background | saved conversation turn")
    except Exception:
        logger.exception("background | failed to save conversation")
        return

    if vector is not None:
        try:
            meta: dict = {
                "type": "conversation",
                "response": response,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            if emotion is not None:
                meta.update(emotion.as_dict())
            await vector.upsert([f"User: {query}\nAssistant: {response}"], [meta])
            logger.info("background | indexed exchange in vector store")
        except Exception:
            logger.exception("background | failed to index exchange")

    try:
        await memory.enqueue_curation(
            query, response,
            emotion=emotion.emotion if emotion else None,
            vad=emotion.vad if emotion else None,
        )
        if curator is not None:
            curator.notify()
    except Exception:
        logger.exception("background | failed to enqueue curation")
