from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from brain.state import BrainState

if TYPE_CHECKING:
    from brain.llm_service import LLMService
    from brain.memory_service import MemoryService

logger = logging.getLogger(__name__)

_FACT_EXTRACTION_PROMPT = """\
Extract at most 3 important, reusable facts from this conversation exchange.
Output each fact as "key: value" on its own line. Skip trivial or context-dependent details.
If there are no useful facts, output NONE.

User: {query}
Assistant: {response}
"""


def make_memory_node(memory: "MemoryService", llm: "LLMService"):
    """Factory: binds memory + LLM, returns the async node callable."""

    async def memory_node(state: BrainState) -> dict:
        query = state["query"]
        response = state.get("response", "")

        try:
            raw = await llm.generate(
                _FACT_EXTRACTION_PROMPT.format(query=query, response=response)
            )
            if raw.strip().upper() != "NONE":
                for line in raw.strip().splitlines():
                    m = re.match(r"^(.+?):\s*(.+)$", line.strip())
                    if m:
                        key, value = m.group(1).strip(), m.group(2).strip()
                        await memory.upsert_fact(key, value)
                        logger.debug("memory_node | upserted fact '%s'", key)
        except Exception as exc:
            logger.warning("memory_node | fact extraction failed: %s", exc)

        return {}

    return memory_node
