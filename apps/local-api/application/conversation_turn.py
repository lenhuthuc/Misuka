"""Shared policy for a single chat turn.

Used by both the buffered (`POST /v1/chat`) and streaming
(`POST /v1/chat/stream`) endpoints, which used to run through two different
implementations of the same three steps — buffered went through a
LangGraph pipeline (`brain.graph.build_fast_graph`), streaming reimplemented
the RAG-decision + retrieval + message-building steps by hand because
LangGraph's node model doesn't stream token-by-token cleanly. That drift is
exactly what REFACTOR_PLAN.md Phase 4 calls out ("tránh hai implementation
lệch nhau") — this module is the single place that policy now lives.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from brain.nodes.generate import build_messages
from brain.nodes.should_rag import should_use_rag

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from brain.memory_service import MemoryService
    from brain.rag_service import RAGService
    from brain.state import RetrievedDoc


@dataclass
class TurnContext:
    """Resolved inputs for the generation step of one chat turn."""

    messages: list[dict[str, str]]
    generated_queries: list[str] = field(default_factory=list)
    retrieved_docs: "list[RetrievedDoc]" = field(default_factory=list)


async def prepare_turn(
    query: str,
    memory: "MemoryService",
    rag: "RAGService",
    recent_limit: int = 10,
    on_rag_error: Callable[[Exception], None] | None = None,
    history_char_budget: int = 3000,
    facts_char_budget: int = 600,
) -> TurnContext:
    """Run the RAG-decision + retrieval + message-building steps shared by
    every chat turn.

    RAG failure degrades to no-context rather than failing the turn — the
    LLM still answers, just without retrieved documents. `on_rag_error` lets
    each endpoint log the failure in its own voice without duplicating the
    try/except here.
    """
    generated_queries: list[str] = []
    retrieved_docs: "list[RetrievedDoc]" = []
    context = ""

    recent = await memory.get_recent(recent_limit)
    # Oldest message the prompt already carries verbatim; anything the retriever
    # finds from at or after this point is a duplicate of the history window.
    covered_since = recent[0]["timestamp"] if recent else None

    if should_use_rag(query):
        try:
            generated_queries, retrieved_docs, context = await rag.build_context(
                query, covered_since=covered_since,
            )
        except Exception as exc:
            if on_rag_error:
                on_rag_error(exc)

    # Facts are what survives beyond the history window and the vector store's
    # per-session churn, so they are fetched for every turn rather than only
    # when retrieval fires.
    facts = await memory.list_facts()
    messages = build_messages(
        query, context, recent, history_char_budget, facts, facts_char_budget,
    )

    # Prefill cost is linear in prompt size and dominates time-to-first-token on
    # a CPU runner, so the prompt budget is a latency number worth watching --
    # not just a context-window concern.
    context_chars = len(context)
    total_chars = sum(len(m["content"]) for m in messages)
    logger.info(
        "turn prompt | messages=%d docs=%d facts=%d context_chars=%d total_chars=%d",
        len(messages), len(retrieved_docs), len(facts), context_chars, total_chars,
    )

    return TurnContext(messages=messages, generated_queries=generated_queries, retrieved_docs=retrieved_docs)
