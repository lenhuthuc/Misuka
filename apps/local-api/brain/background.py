"""Post-response background tasks: save conversation turn, decide whether to extract facts."""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from brain.emotion_service import EmotionReading
    from brain.llm_service import LLMService
    from brain.memory_service import MemoryService
    from brain.vector_service import VectorService

logger = logging.getLogger(__name__)

_DECIDE_PROMPT = """\
Does the following conversation contain facts worth remembering long-term?
Worth remembering: user preferences, named entities, system config, technical facts.
NOT worth remembering: greetings, simple yes/no, transient context.

User: {query}
Assistant: {response}

Answer YES or NO only."""

_FACT_EXTRACTION_PROMPT = """\
Extract at most 3 important, reusable facts from this conversation exchange.
Output each fact as "key: value" on its own line. Skip trivial or context-dependent details.
If there are no useful facts, output NONE.

User: {query}
Assistant: {response}
"""


async def run_memory_tasks(
    query: str,
    response: str,
    memory: "MemoryService",
    llm: "LLMService",
    emotion: "EmotionReading | None" = None,
    vector: "VectorService | None" = None,
) -> None:
    """Save conversation turn (with the response's V/A/D + emotion), index the
    exchange as a retrievable memory, and optionally extract long-term facts.

    Runs entirely after the HTTP response has been returned to the client —
    latency here does not affect the user.
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

    # Index the exchange in Qdrant so RAG retrieval returns the response along
    # with its V/A/D + emotion label.
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
        raw = await llm.generate(_DECIDE_PROMPT.format(query=query, response=response))
        should_save = raw.strip().upper().startswith("YES")
    except Exception:
        logger.exception("background | decide LLM failed, skipping fact extraction")
        return

    if not should_save:
        return

    try:
        raw = await llm.generate(_FACT_EXTRACTION_PROMPT.format(query=query, response=response))
        if raw.strip().upper() != "NONE":
            for line in raw.strip().splitlines():
                m = re.match(r"^(.+?):\s*(.+)$", line.strip())
                if m:
                    key, value = m.group(1).strip(), m.group(2).strip()
                    await memory.upsert_fact(key, value)
                    logger.debug("background | upserted fact '%s'", key)
    except Exception:
        logger.exception("background | fact extraction failed")
