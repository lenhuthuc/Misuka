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

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from brain.nodes.generate import build_messages
from brain.nodes.should_rag import should_use_rag

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

    if should_use_rag(query):
        try:
            generated_queries, retrieved_docs, context = await rag.build_context(query)
        except Exception as exc:
            if on_rag_error:
                on_rag_error(exc)

    messages = await build_messages(query, context, memory, recent_limit)
    return TurnContext(messages=messages, generated_queries=generated_queries, retrieved_docs=retrieved_docs)
