"""Tests for the two things that keep a chat turn's prompt from growing without
bound: dropping retrieved turns the history window already spells out, and
capping retrieved context by character budget.

ROOT CAUSE these tests pin down:

Every exchange is indexed into the vector store as it happens, while the prompt
separately carries the last N messages verbatim. Retrieval therefore handed
back the very turns the history already contained, and the stored responses
grew with each turn. Measured over a real five-turn conversation the prompt
went 4,493 -> 15,131 characters and time-to-first-token 11.3s -> 43.8s, because
prefill cost is linear in prompt length on a CPU runner.

  before: build_context(query) -> every hit, whole content, no ceiling
  after:  build_context(query, covered_since=<oldest history timestamp>)
          -> duplicates dropped, remainder trimmed to the character budget
"""
from __future__ import annotations

import pytest

from brain.rag_service import RAGService


class StubHit:
    """Minimal stand-in for a Qdrant hit: an id plus a payload."""

    def __init__(self, hit_id: str, content: str, timestamp: str | None = None) -> None:
        self.id = hit_id
        self.payload: dict = {"content": content}
        if timestamp is not None:
            self.payload["timestamp"] = timestamp


class StubVectorService:
    def __init__(self, hits: list[StubHit]) -> None:
        self._hits = hits

    async def search_batch(self, queries: list[str], top_k: int) -> list[list[StubHit]]:
        return [self._hits[:top_k] for _ in queries]


def make_rag(hits: list[StubHit], budget: int = 2000) -> RAGService:
    return RAGService(vector=StubVectorService(hits), top_k=10, context_char_budget=budget)


@pytest.mark.asyncio
async def test_docs_from_inside_the_history_window_are_dropped():
    """@example: a doc indexed after the oldest history message is already in the
    prompt verbatim -> it must not be pasted in again as context."""
    rag = make_rag([
        StubHit("old", "a genuinely older memory", timestamp="2026-08-01T00:00:00+00:00"),
        StubHit("dup", "this turn is in the history window", timestamp="2026-08-09T02:30:00+00:00"),
    ])

    _queries, docs, context = await rag.build_context(
        "tell me about my practice", covered_since="2026-08-09T02:00:00+00:00",
    )

    assert [d["doc_id"] for d in docs] == ["old"]
    assert "this turn is in the history window" not in context
    assert "a genuinely older memory" in context


@pytest.mark.asyncio
async def test_doc_exactly_at_the_window_boundary_counts_as_covered():
    """@example: timestamp equal to the oldest history message -> that message IS
    the history window's first entry, so it is a duplicate."""
    rag = make_rag([StubHit("edge", "boundary turn", timestamp="2026-08-09T02:00:00+00:00")])

    _queries, docs, _context = await rag.build_context(
        "anything", covered_since="2026-08-09T02:00:00+00:00",
    )

    assert docs == []


@pytest.mark.asyncio
async def test_docs_without_a_timestamp_survive_filtering():
    """@example: seeded knowledge documents carry no timestamp -> they are not
    conversation turns and must never be filtered as duplicates."""
    rag = make_rag([StubHit("seed", "seeded knowledge with no timestamp")])

    _queries, docs, _context = await rag.build_context(
        "anything", covered_since="2026-08-09T02:00:00+00:00",
    )

    assert [d["doc_id"] for d in docs] == ["seed"]


@pytest.mark.asyncio
async def test_context_is_trimmed_to_the_character_budget():
    """@example: three 400-char docs against a 900-char budget -> only the two
    that fit are kept, in rank order."""
    rag = make_rag(
        [StubHit(f"d{i}", "x" * 400) for i in range(3)],
        budget=900,
    )

    _queries, docs, _context = await rag.build_context("anything")

    assert [d["doc_id"] for d in docs] == ["d0", "d1"]


@pytest.mark.asyncio
async def test_a_single_oversized_doc_is_still_returned():
    """@example: the top hit alone exceeds the budget -> keep it rather than
    return empty context, so the budget never starves retrieval entirely."""
    rag = make_rag([StubHit("huge", "y" * 5000)], budget=900)

    _queries, docs, context = await rag.build_context("anything")

    assert [d["doc_id"] for d in docs] == ["huge"]
    assert "y" * 5000 in context


@pytest.mark.asyncio
async def test_without_covered_since_nothing_is_filtered_as_duplicate():
    """@example: an empty conversation has no history window -> retrieval keeps
    every hit it found."""
    rag = make_rag([
        StubHit("a", "first", timestamp="2026-08-09T02:30:00+00:00"),
        StubHit("b", "second", timestamp="2026-08-09T02:31:00+00:00"),
    ])

    _queries, docs, _context = await rag.build_context("anything")

    assert sorted(d["doc_id"] for d in docs) == ["a", "b"]
