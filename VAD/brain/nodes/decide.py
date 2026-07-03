from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from brain.state import BrainState

if TYPE_CHECKING:
    from brain.llm_service import LLMService
    from brain.memory_service import MemoryService

logger = logging.getLogger(__name__)

_DECIDE_PROMPT = """\
Does the following conversation contain facts worth remembering long-term?
Worth remembering: user preferences, named entities, system config, technical facts.
NOT worth remembering: greetings, simple yes/no, transient context.

User: {query}
Assistant: {response}

Answer YES or NO only."""


def make_decide_node(memory: "MemoryService", llm: "LLMService"):
    async def decide_node(state: BrainState) -> dict:
        query = state["query"]
        response = state.get("response", "")

        await memory.save_message("user", query)
        await memory.save_message("assistant", response)
        logger.info("decide_node | saved conversation turn")

        try:
            raw = await llm.generate(
                _DECIDE_PROMPT.format(query=query, response=response)
            )
            should_save = raw.strip().upper().startswith("YES")
        except Exception as exc:
            logger.warning("decide_node | LLM decide failed, skipping facts: %s", exc)
            should_save = False

        logger.info("decide_node | should_save_facts=%s", should_save)
        return {"should_save_facts": should_save}

    return decide_node
