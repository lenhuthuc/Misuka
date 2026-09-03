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

from brain.nodes.generate import build_messages, fit_history_rows
from brain.nodes.should_rag import is_delegating_choice, should_use_rag
from brain.nodes.should_search_web import decide_web_search
from brain.reasoning_policy import ReasoningPolicy, derive_reasoning_policy, history_relevance
from brain.rag_service import RetrievalEvidence
from brain.response_policy import ResponsePolicy

logger = logging.getLogger(__name__)

# Distinguishes live web results from the RAG header `generate._CONTEXT_HEADER`
# already puts around retrieved memories -- both land in the same `context`
# string, and the model should know which is which before trusting either.
_WEB_CONTEXT_HEADER = "Kết quả tìm kiếm internet (có thể chưa hoàn toàn cập nhật):"

if TYPE_CHECKING:
    from brain.memory_service import MemoryService
    from brain.rag_service import RAGService
    from brain.state import RetrievedDoc
    from brain.web_search_service import WebSearchService


@dataclass
class TurnContext:
    """Resolved inputs for the generation step of one chat turn."""

    messages: list[dict[str, str]]
    generated_queries: list[str] = field(default_factory=list)
    retrieved_docs: "list[RetrievedDoc]" = field(default_factory=list)
    web_results: list[dict] = field(default_factory=list)
    reasoning: ReasoningPolicy = field(default_factory=ReasoningPolicy)
    # The VAD policy the caller passed in, possibly retuned for a grounded
    # knowledge turn. Returned rather than left to the caller because whether
    # this turn is grounded is only known *after* retrieval has run, and the
    # caller derived its policy before calling in.
    response_policy: ResponsePolicy = field(default_factory=ResponsePolicy)
    web_search_kind: str = ""
    # The prompt asks the model not to end this turn with a question; this is
    # what lets the caller enforce it after generation, where it is exact.
    # Prompting alone reached 2 clean replies in 3 on these weights.
    forbids_questions: bool = False


async def prepare_turn(
    query: str,
    memory: "MemoryService",
    rag: "RAGService",
    recent_limit: int = 18,
    on_rag_error: Callable[[Exception], None] | None = None,
    history_char_budget: int = 3000,
    response_policy: "ResponsePolicy | None" = None,
    reasoning_enabled: bool = True,
    reasoning_activation_threshold: float = 0.50,
    reasoning_min_tokens: int = 64,
    reasoning_max_tokens: int = 192,
    reasoning_token_scale: float = 0.65,
    web_search: "WebSearchService | None" = None,
    web_search_enabled: bool = True,
    web_search_knowledge_enabled: bool = True,
    knowledge_temperature: float = 0.30,
    knowledge_max_tokens: int = 480,
    on_web_search_error: Callable[[Exception], None] | None = None,
) -> TurnContext:
    """Run the RAG-decision + retrieval + message-building steps shared by
    every chat turn.

    RAG failure degrades to no-context rather than failing the turn — the
    LLM still answers, just without retrieved documents. `on_rag_error` lets
    each endpoint log the failure in its own voice without duplicating the
    try/except here. Web search follows the same degrade-on-failure contract.
    """
    generated_queries: list[str] = []
    retrieved_docs: "list[RetrievedDoc]" = []
    web_results: list[dict] = []
    web_context = ""
    context = ""
    evidence = RetrievalEvidence()

    recent = await memory.get_recent(recent_limit)
    filter_history = getattr(memory, "filter_repetitive_history", None)
    if filter_history is not None:
        recent = filter_history(recent)
    prompt_recent = fit_history_rows(recent, history_char_budget)
    # Oldest message the prompt already carries verbatim; anything the retriever
    # finds from at or after this point is a duplicate of the history window.
    covered_since = prompt_recent[0]["timestamp"] if prompt_recent else None
    recent_history_score = history_relevance(query, prompt_recent)

    # Strict latency order: use the verbatim recent window first. Only pay for
    # vector retrieval when those latest turns do not already explain the new
    # input; only after both remain weak may adaptive reasoning run.
    rag_eligible = should_use_rag(query)
    if rag_eligible and recent_history_score < reasoning_activation_threshold:
        try:
            build_with_evidence = getattr(rag, "build_context_with_evidence", None)
            if build_with_evidence is not None:
                generated_queries, retrieved_docs, context, evidence = await build_with_evidence(
                    query, covered_since=covered_since,
                )
            else:
                generated_queries, retrieved_docs, context = await rag.build_context(
                    query, covered_since=covered_since,
                )
        except Exception as exc:
            if on_rag_error:
                on_rag_error(exc)
    elif rag_eligible:
        logger.info(
            "turn retrieval | skipped=recent_history history_score=%.3f threshold=%.3f",
            recent_history_score, reasoning_activation_threshold,
        )

    # Independent of the RAG gate above: RAG answers "what did we already
    # talk about", this answers "what is true about the world right now" or
    # "what is true about the world at all", so a query can trigger both,
    # either, or neither.
    #
    # `prompt_recent` is passed so a topic-free turn can inherit the subject
    # it is continuing. The utterance that most needs grounding is often the
    # one with the least text in it -- "ok bạn cứ chọn" is the whole question,
    # and the thing being chosen is two messages back.
    web_decision = decide_web_search(
        query, prompt_recent, knowledge_enabled=web_search_knowledge_enabled,
    )
    if web_search_enabled and web_search is not None and web_decision.should_search:
        try:
            web_results, web_context = await web_search.build_context(web_decision.query)
        except Exception as exc:
            if on_web_search_error:
                on_web_search_error(exc)
    if web_context:
        context = f"{context}\n\n{_WEB_CONTEXT_HEADER}\n{web_context}" if context else f"{_WEB_CONTEXT_HEADER}\n{web_context}"

    reasoning = derive_reasoning_policy(
        query,
        prompt_recent,
        rag_score=evidence.rag_score,
        indexed_history_score=evidence.history_score,
        activation_threshold=reasoning_activation_threshold,
        min_tokens=reasoning_min_tokens,
        max_tokens=reasoning_max_tokens,
        token_scale=reasoning_token_scale,
        enabled=reasoning_enabled,
    )

    # Grounded, not merely *asked*: the treatment is earned by having results in
    # hand. A question whose search returned nothing gets the ordinary chat
    # decoding, because a colder, longer answer with no notes behind it is just
    # a more confident invention.
    #
    # Which *gate* fetched them is a separate question from whether they are
    # there, and conflating the two was a bug: a "live" turn used to get its
    # snippets pasted into the prompt with no instruction to use them and
    # chat-temperature decoding, so the model talked past them. Both kinds now
    # get the grounding treatment; only a knowledge turn also gets the 3-5
    # sentence length override, because a live answer is a fact or two.
    grounded_web = bool(web_context)
    grounded_knowledge = web_decision.kind == "knowledge" and grounded_web
    policy = response_policy or ResponsePolicy()
    if grounded_knowledge:
        policy = policy.for_grounded_knowledge(knowledge_temperature, knowledge_max_tokens)
    elif grounded_web:
        policy = policy.for_grounded_web(knowledge_temperature)

    messages = build_messages(
        query, context, recent, history_char_budget,
        response_policy_instruction=policy.instruction,
        grounded_knowledge=grounded_knowledge,
        grounded_web=grounded_web,
    )

    # Prefill cost is linear in prompt size and dominates time-to-first-token on
    # a CPU runner, so the prompt budget is a latency number worth watching --
    # not just a context-window concern.
    context_chars = len(context)
    total_chars = sum(len(m["content"]) for m in messages)
    logger.info(
        "turn prompt | messages=%d docs=%d web_results=%d context_chars=%d total_chars=%d "
        "thinking=%s thinking_budget=%d context_confidence=%.3f rag_score=%.3f history_score=%.3f "
        "web=%s web_inherited=%s grounded_web=%s grounded_knowledge=%s",
        len(messages), len(retrieved_docs), len(web_results), context_chars, total_chars,
        reasoning.enabled, reasoning.token_budget, reasoning.confidence,
        reasoning.rag_score, reasoning.history_score,
        web_decision.kind or "none", web_decision.inherited, grounded_web, grounded_knowledge,
    )

    return TurnContext(
        messages=messages,
        generated_queries=generated_queries,
        retrieved_docs=retrieved_docs,
        web_results=web_results,
        reasoning=reasoning,
        response_policy=policy,
        web_search_kind=web_decision.kind,
        forbids_questions=is_delegating_choice(query),
    )
