"""Tests for the post-response work of a chat turn.

ROOT CAUSE these pin down:

`run_memory_tasks` was spawned the instant the last token was generated and
went through no gate at all -- the `LLMPriorityGate` only ever guarded LLM
callers. Its `vector.upsert` runs an embedding, and on this CPU that is ~3s of
exactly the cores Piper needs to render the reply. Observed in the server log:

    15:48:23  background | saved conversation turn
    15:48:26  background | indexed exchange in vector store
    15:48:26  POST /v1/audio/speech 200 OK

The user waited through the embedding before hearing a word. The two sqlite
writes stay immediate (a millisecond each, and the next turn's history window
reads them); the embedding waits until the reply has been spoken.
"""
from __future__ import annotations

import asyncio

import pytest

from brain.background import run_memory_tasks
from core.llm_priority import LLMPriorityGate
from tests.conftest import FakeMemoryService, FakeVectorService


@pytest.mark.asyncio
async def test_the_exchange_is_saved_before_the_reply_is_spoken():
    """@example: the turn ends -> the conversation rows exist immediately, so a
    user who answers straight away still finds the exchange in the history
    window the next turn builds its prompt from."""
    memory, vector = FakeMemoryService(), FakeVectorService()
    gate = LLMPriorityGate()
    gate.mark_active()

    task = asyncio.create_task(run_memory_tasks(
        "câu hỏi", "câu trả lời", memory, vector=vector, gate=gate, defer_timeout=5.0,
    ))
    await asyncio.sleep(0.05)

    assert [m["role"] for m in memory.messages] == ["user", "assistant"]
    assert vector.upserted == []

    gate.speech_finished()
    await task


@pytest.mark.asyncio
async def test_indexing_waits_for_the_reply_to_be_spoken():
    """@example: the client reports its playback queue drained -> the embedding
    runs then, not while Piper is still rendering the same reply."""
    memory, vector = FakeMemoryService(), FakeVectorService()
    gate = LLMPriorityGate()
    gate.mark_active()

    task = asyncio.create_task(run_memory_tasks(
        "câu hỏi", "câu trả lời", memory, vector=vector, gate=gate, defer_timeout=5.0,
    ))
    await asyncio.sleep(0.05)
    assert vector.upserted == []

    gate.speech_finished()
    await task

    assert len(vector.upserted) == 1
    assert vector.upserted[0][0] == ["User: câu hỏi\nAssistant: câu trả lời"]


@pytest.mark.asyncio
async def test_indexing_still_happens_when_the_signal_never_arrives():
    """@example: the page was closed mid-reply -> the exchange is indexed late
    rather than dropped out of retrieval for the rest of the session."""
    memory, vector = FakeMemoryService(), FakeVectorService()
    gate = LLMPriorityGate()
    gate.mark_active()

    await run_memory_tasks(
        "câu hỏi", "câu trả lời", memory, vector=vector, gate=gate, defer_timeout=0.1,
    )

    assert len(vector.upserted) == 1


@pytest.mark.asyncio
async def test_without_a_gate_everything_runs_straight_through():
    """@example: a caller with no gate (the buffered path in a test, a script) ->
    the pre-existing behaviour, with nothing to wait on."""
    memory, vector = FakeMemoryService(), FakeVectorService()

    await run_memory_tasks("câu hỏi", "câu trả lời", memory, vector=vector)

    assert len(memory.messages) == 2
    assert len(vector.upserted) == 1
