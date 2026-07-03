from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from brain.state import BrainState

if TYPE_CHECKING:
    from brain.rag_service import RAGService

logger = logging.getLogger(__name__)


def make_retrieve_node(rag: "RAGService"):
    """Factory: binds a RAGService instance, returns the async node callable."""

    async def retrieve_node(state: BrainState) -> dict:
        query = state["query"]
        logger.info("retrieve_node | query=%r", query)
        generated_queries, retrieved_docs, context = await rag.build_context(query)
        return {
            "generated_queries": generated_queries,
            "retrieved_docs": retrieved_docs,
            "context": context,
        }

    return retrieve_node
