"""The retriever's second gate: a similarity floor on the raw hits.

`should_use_rag` decides whether a *query* can benefit from retrieval. It
cannot decide whether the store actually holds anything relevant, and Qdrant
answers every search with its `top_k` nearest points however far away they
are -- so an unrelated question still came back with five memories and spent
them on prefill. Measured on a real turn: `docs=3 context_chars=886` of
conversation history that had nothing to do with what was asked.

The floor has to be applied before fusion. RRF scores a hit by its *rank*
among the others, so after fusion "the best of a bad lot" and "relevant" are
indistinguishable.
"""
from __future__ import annotations

import pytest

from brain.rag_service import RAGService


class StubHit:
    def __init__(self, hit_id: str, content: str, score: float, timestamp: str | None = None) -> None:
        self.id = hit_id
        self.score = score
        self.payload: dict = {"content": content}
        if timestamp is not None:
            self.payload["timestamp"] = timestamp


class StubVectorService:
    def __init__(self, hits: list[StubHit]) -> None:
        self._hits = hits

    async def search_batch(self, queries: list[str], top_k: int) -> list[list[StubHit]]:
        return [self._hits[:top_k] for _ in queries]


@pytest.mark.asyncio
async def test_hits_below_the_floor_never_reach_the_prompt():
    """@example: nothing in the store is about the question -> no context block,
    and the prompt stays the size it would have been without retrieval."""
    vector = StubVectorService([
        StubHit("a", "hôm qua mình đi biển", 0.11),
        StubHit("b", "mình thích ăn phở", 0.09),
    ])
    rag = RAGService(vector=vector, min_score=0.35)  # type: ignore[arg-type]

    _, docs, context = await rag.build_context("một cộng một bằng mấy")

    assert docs == []
    assert context == ""


@pytest.mark.asyncio
async def test_a_genuinely_relevant_hit_still_gets_through():
    """@example: the question is about something the store remembers -> the floor
    is a filter on irrelevance, not a switch that turns RAG off."""
    vector = StubVectorService([
        StubHit("a", "mình thích ăn phở", 0.71),
        StubHit("b", "hôm qua trời mưa", 0.12),
    ])
    rag = RAGService(vector=vector, min_score=0.35)  # type: ignore[arg-type]

    _, docs, context = await rag.build_context("mình thích ăn gì")

    assert [d["doc_id"] for d in docs] == ["a"]
    assert "phở" in context


@pytest.mark.asyncio
async def test_the_floor_is_off_by_default():
    """@example: a caller that did not opt in -> every hit is kept, which is the
    behaviour every existing test and the seeded-document path rely on."""
    vector = StubVectorService([StubHit("a", "bất kỳ", 0.01)])
    rag = RAGService(vector=vector)  # type: ignore[arg-type]

    _, docs, _ = await rag.build_context("câu hỏi nào đó")

    assert len(docs) == 1


@pytest.mark.asyncio
async def test_raw_scores_are_split_between_recent_history_and_long_term_rag():
    vector = StubVectorService([
        StubHit("recent", "recent exchange", 0.78, "2026-08-09T02:30:00+00:00"),
        StubHit("old", "older memory", 0.31, "2026-08-01T00:00:00+00:00"),
    ])
    rag = RAGService(vector=vector, min_score=0.50)  # type: ignore[arg-type]

    _, docs, _, evidence = await rag.build_context_with_evidence(
        "question", covered_since="2026-08-09T02:00:00+00:00",
    )

    assert docs == []
    assert evidence.history_score == pytest.approx(0.78)
    assert evidence.rag_score == pytest.approx(0.31)
