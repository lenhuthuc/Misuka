"""Tests for the background worker that mines finished exchanges for durable facts.

ROOT CAUSE this design addresses:

Facts are the only memory that outlives both the history window and the
in-memory vector store, but extracting them costs an LLM call, and on this CPU
the runner is serialised. Doing it per-turn as a fire-and-forget task queued it
in front of the user's next turn, pushing time-to-first-token from 3.3s to
16-19s; gating it without persistence meant every attempt during an active
conversation was abandoned and lost, so across a five-turn measurement nothing
was ever extracted and the facts table stayed empty.

  before: run_memory_tasks -> 2 ungated llm.generate calls, lost if abandoned
  after:  run_memory_tasks -> durable queue row; MemoryCurator batches the
          queue through LLMPriorityGate and retries what it could not finish

An earlier version of this worker also distilled each exchange into a compact
note. It was removed after measurement rather than tuned: see the module
docstring in `brain.curator` for what was tried and why the question itself was
wrong.
"""
from __future__ import annotations

import asyncio

import pytest

from brain.curator import MemoryCurator, _grounding_corpus, _parse_facts
from core.llm_priority import LLMPriorityGate

from tests.conftest import FakeLLMService, FakeMemoryService


def make_curator(reply: str, batch_size: int = 5):
    memory = FakeMemoryService()
    llm = FakeLLMService(generate_reply=reply)
    gate = LLMPriorityGate(settle_seconds=0.0)
    curator = MemoryCurator(memory=memory, llm=llm, gate=gate, batch_size=batch_size)
    return curator, memory, llm, gate


# ── parsing ──────────────────────────────────────────────────────────────────
#
# Observed in a live run: the model ignored the format and kept writing prose,
# and every prose line with a colon was stored as a fact. The table filled with
# markdown headings, numbered list items, and the prompt's own placeholder --
# 16 junk rows and not one usable fact.

def test_a_genuine_short_fact_survives():
    """@example: a real identifier-shaped fact -> kept."""
    assert _parse_facts("practice_time: evening") == [("practice_time", "evening")]


def test_several_facts_are_read_in_order():
    """@example: the shape 1.5b actually returns -> all parsed."""
    raw = "name: Thuc\nlanguage: Vietnamese\nlevel: Intermediate (IELTS 6.0)"

    assert _parse_facts(raw) == [
        ("name", "Thuc"),
        ("language", "Vietnamese"),
        ("level", "Intermediate (IELTS 6.0)"),
    ]


def test_numbered_and_bolded_facts_are_still_read():
    """@example: the shape qwen2.5:1.5b actually returns, despite the prompt
    forbidding it -> the decoration is stripped and the facts kept.

    A live run extracted all six facts correctly and every one was discarded,
    purely because they arrived as "1. **Name: Thuc**"."""
    raw = "1. **Name: Thuc**\n2. **Language: Vietnamese**\n3. **Schedule: After 9pm**"

    assert _parse_facts(raw) == [
        ("name", "Thuc"),
        ("language", "Vietnamese"),
        ("schedule", "After 9pm"),
    ]


def test_bulleted_facts_are_still_read():
    """@example: the same prompt, a later run, bullets instead of numbers ->
    also stripped, so the key is "Name" and not "- **Name"."""
    raw = "- **Name: Thuc**\n- **Language: Vietnamese**"

    assert _parse_facts(raw) == [("name", "Thuc"), ("language", "Vietnamese")]


def test_keys_are_lowercased_so_they_cannot_duplicate():
    """@example: "Name" one run and "name" the next -> both normalise to the same
    key, so `upsert_fact` updates one fact instead of asserting two."""
    assert _parse_facts("Name: Thuc") == _parse_facts("name: Thuc") == [("name", "Thuc")]


def test_prose_lines_are_not_mistaken_for_facts():
    """@example: an instruction sentence containing a colon -> rejected.

    Stripping decoration means the leading "3." no longer disqualifies this on
    its own, so the key word count is what has to hold the line: a three-word
    key is a heading, not an identifier."""
    raw = "3. Main Topic Discussion: Focus on the chosen topic and discuss it in detail."

    assert _parse_facts(raw) == []


def test_long_values_are_not_facts():
    """@example: a short key with a paragraph attached -> rejected, since a
    stored fact is a value and not an instruction."""
    raw = "structure: " + "The session is divided into an introduction and a warm-up " * 3

    assert _parse_facts(raw) == []


def test_two_word_keys_are_allowed():
    """@example: "IELTS band" -> kept; real keys run to two words, and the
    ceiling must not be so tight that it rejects them.

    The value here also guards a decoration-stripping bug: "6.0" begins with a
    digit followed by a dot, exactly like list numbering, and an earlier version
    stripped it down to "0"."""
    assert _parse_facts("IELTS band: 6.0") == [("ielts band", "6.0")]


def test_decoration_stripping_leaves_decimal_values_intact():
    """@example: a bare decimal value -> preserved, not read as "1." numbering."""
    assert _parse_facts("score: 7.5") == [("score", "7.5")]


def test_a_speaker_prefix_does_not_become_the_key():
    """@example: the model prefixes the fact with the user's name -> the real
    key is used, not "Thuc".

    Observed live: "Thuc: persistent difficulties: Pronunciation of /th/ sounds"
    was stored under the key "Thuc"."""
    raw = "Thuc: persistent difficulties: Pronunciation of /th/ sounds"

    assert _parse_facts(raw) == [("persistent difficulties", "Pronunciation of /th/ sounds")]


def test_a_value_that_merely_contains_a_colon_is_left_alone():
    """@example: a clock time inside the value -> not mistaken for a nested key,
    since "9" does not look like one."""
    assert _parse_facts("practice_time: 9:00 PM") == [("practice_time", "9:00 PM")]


def test_prompt_placeholders_are_not_facts():
    """@example: the model echoes the template's own "key: value" -> rejected."""
    assert _parse_facts("key: value") == []


def test_fact_count_is_capped_at_what_the_prompt_asked_for():
    """@example: the model emits ten well-formed facts -> only the ceiling is kept."""
    assert len(_parse_facts("\n".join(f"k{i}: v{i}" for i in range(10)))) == 6


def test_none_reply_yields_nothing():
    """@example: the model correctly reports no durable facts -> empty list, and
    "NONE" is never itself stored as a fact."""
    assert _parse_facts("NONE") == []


def test_unstructured_reply_yields_nothing():
    """@example: the model ignores the format entirely -> nothing is stored,
    rather than prose being written into the facts table."""
    assert _parse_facts("I am not sure what you want from me here.") == []


# ── grounding ────────────────────────────────────────────────────────────────
#
# Observed live: a batch holding a single exchange about written feedback still
# produced six facts, including "name: User" and "language: English" for a user
# who had said he was Thuc and Vietnamese. Because `upsert_fact` overwrites by
# key, that invented name replaced the correct one extracted from an earlier,
# better-informed batch — and then went into every later system prompt.

def test_invented_facts_are_dropped():
    """@example: the model offers an age the user never mentioned -> discarded,
    because a wrong fact misinforms every later turn while a missing one only
    costs recall once."""
    corpus = _grounding_corpus([{"query": "My name is Thuc.", "response": "Nice to meet you."}])

    assert _parse_facts("name: Thuc\nage: 25", corpus) == [("name", "Thuc")]


def test_the_speaker_label_cannot_become_the_users_name():
    """@example: "name: User", lifted from the prompt's own speaker label ->
    dropped, since the grounding corpus excludes prompt scaffolding."""
    corpus = _grounding_corpus([{"query": "I practise late.", "response": "Noted."}])

    assert _parse_facts("name: User", corpus) == []


def test_grounded_facts_survive():
    """@example: every value echoes something the user actually said -> all kept."""
    corpus = _grounding_corpus([{
        "query": "I am Vietnamese and I practise after 9pm.",
        "response": "Understood.",
    }])

    facts = _parse_facts("language: Vietnamese\nschedule: after 9pm", corpus)

    assert facts == [("language", "Vietnamese"), ("schedule", "after 9pm")]


def test_grounding_is_skipped_when_no_corpus_is_given():
    """@example: parsing without a corpus -> no grounding applied, which is what
    the pure parsing tests above rely on."""
    assert _parse_facts("age: 25") == [("age", "25")]


# ── batching and storage ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_whole_batch_is_mined_in_a_single_llm_call():
    """@example: three queued exchanges -> exactly one generate call, because
    batching is what amortises prefill on a bandwidth-bound runner."""
    curator, memory, llm, _gate = make_curator("name: Thuc")
    for i in range(3):
        await memory.enqueue_curation(f"q{i}", f"r{i}")

    assert await curator._process_batch() is True

    assert len(llm.generate_calls) == 1
    assert memory.curation_queue == []


@pytest.mark.asyncio
async def test_extracted_facts_are_persisted():
    """@example: the reply carries facts the exchange supports -> they land in
    the facts table, which is what later turns read into the system prompt."""
    curator, memory, _llm, _gate = make_curator("level: intermediate\ngoal: fluency")
    await memory.enqueue_curation(
        "I am at an intermediate level and my goal is fluency.", "Noted.",
    )

    await curator._process_batch()

    assert memory.facts == {"level": "intermediate", "goal": "fluency"}


@pytest.mark.asyncio
async def test_ungrounded_facts_never_reach_the_table():
    """@example: the model invents a fact the batch never mentioned -> it is
    dropped before storage, so it can never overwrite a correct one."""
    curator, memory, _llm, _gate = make_curator("name: Thuc\ngender: Female")
    await memory.enqueue_curation("My name is Thuc.", "Nice to meet you.")

    await curator._process_batch()

    assert memory.facts == {"name": "Thuc"}


@pytest.mark.asyncio
async def test_batch_clears_even_when_it_holds_no_facts():
    """@example: a batch of small talk -> queue drains anyway, so exchanges
    without facts cannot pile up and block the ones behind them."""
    curator, memory, _llm, _gate = make_curator("NONE")
    await memory.enqueue_curation("hi", "hello")

    assert await curator._process_batch() is True

    assert memory.curation_queue == []
    assert memory.facts == {}


@pytest.mark.asyncio
async def test_every_queued_exchange_reaches_the_prompt():
    """@example: a batch of two -> both exchanges appear in the single call, so
    batching never silently drops the material it was meant to mine."""
    curator, memory, llm, _gate = make_curator("NONE")
    await memory.enqueue_curation("what is my name", "you are Thuc")
    await memory.enqueue_curation("when do I practise", "after 9pm")

    await curator._process_batch()

    prompt = llm.generate_calls[0]
    assert "what is my name" in prompt
    assert "when do I practise" in prompt


# ── durability under interruption ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_abandoned_batch_stays_queued_for_a_later_attempt():
    """@example: a turn starts while the curator is mid-generation -> the work is
    dropped but the queue row survives, which is the whole point of persisting
    it. The previous fire-and-forget design lost this work permanently."""
    curator, memory, llm, gate = make_curator("name: Thuc")
    await memory.enqueue_curation("q", "r")

    async def slow_generate(prompt: str) -> str:
        await asyncio.sleep(5.0)
        return "name: Thuc"

    llm.generate = slow_generate  # type: ignore[method-assign]

    runner = asyncio.create_task(curator._process_batch())
    await asyncio.sleep(0.05)

    async with gate.foreground():
        completed = await runner
        await asyncio.sleep(0.01)

    assert completed is False
    assert len(memory.curation_queue) == 1
    assert memory.facts == {}


@pytest.mark.asyncio
async def test_a_failing_batch_stops_being_retried_forever():
    """@example: the LLM keeps raising -> attempts are counted, and once they hit
    the ceiling the batch is no longer eligible, so one poisonous row cannot
    block the queue indefinitely."""
    curator, memory, llm, _gate = make_curator("unused")
    await memory.enqueue_curation("q", "r")

    async def failing_generate(prompt: str) -> str:
        raise RuntimeError("curator model unavailable")

    llm.generate = failing_generate  # type: ignore[method-assign]

    for _ in range(3):
        with pytest.raises(RuntimeError):
            await curator._process_batch()

    assert await memory.pending_curation_count() == 0
    assert len(memory.curation_queue) == 1  # retained for inspection, not retried


@pytest.mark.asyncio
async def test_empty_queue_is_a_no_op():
    """@example: nothing pending -> no LLM call at all."""
    curator, _memory, llm, _gate = make_curator("unused")

    assert await curator._process_batch() is True
    assert llm.generate_calls == []
