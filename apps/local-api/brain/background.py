"""Post-response work for a chat turn: persist it and index it for retrieval.

Everything here runs after the HTTP response has been returned, and — by
design — none of it calls the LLM. On this CPU the runner is serialised, so an
LLM call issued here would land in front of the user's next turn. Durable
memory past the history window is the vector store's job, and embedding is the
only heavy step left.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from brain.emotion_service import EmotionReading
    from brain.memory_service import MemoryService
    from brain.vector_service import VectorService
    from core.llm_priority import LLMPriorityGate

logger = logging.getLogger(__name__)


async def run_memory_tasks(
    query: str,
    response: str,
    memory: "MemoryService",
    emotion: "EmotionReading | None" = None,
    vector: "VectorService | None" = None,
    gate: "LLMPriorityGate | None" = None,
    defer_timeout: float = 90.0,
    session_id: str = "default",
) -> None:
    """Save the turn and index it for retrieval.

    Use when: a chat turn has finished and its response has been sent.

    Expects: to be spawned as a background task, never awaited by a request.

    The two sqlite writes happen immediately. They cost a millisecond each, and
    the conversation row has to be there before the *next* turn reads its
    history window -- deferring it would mean an exchange that could vanish
    from the prompt if the user answered quickly.

    Indexing is the opposite: `vector.upsert` runs an embedding, and measured
    on this CPU that is ~3s of the same cores Piper needs to render the reply.
    Spawned at the end of generation, it lands exactly on top of synthesis and
    the user waits longer to hear anything. So it waits behind `gate` until the
    reply has actually been spoken (see core/llm_priority.py); without a gate
    it runs straight away, which is the old behaviour.

    """
    try:
        await memory.save_message("user", query, session_id=session_id)
        await memory.save_message(
            "assistant", response,
            vad=emotion.vad if emotion else None,
            emotion=emotion.emotion if emotion else None,
            session_id=session_id,
        )
        logger.info("background | saved conversation turn")
    except Exception:
        logger.exception("background | failed to save conversation")
        return

    if vector is not None:
        if gate is not None and not await gate.wait_until_spoken(timeout=defer_timeout):
            # The client never reported its playback queue draining and no
            # reservation covered it either -- index anyway rather than lose
            # the exchange from retrieval for the rest of the session.
            logger.info("background | no end-of-speech signal in %.0fs, indexing anyway", defer_timeout)

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
