"""Tests for putting stored facts into the system prompt.

ROOT CAUSE this closes:

`MemoryService.upsert_fact` had been writing to the facts table since the
project started, but `list_facts` and `get_fact` were never called from
anywhere. The whole fact-extraction branch — the LLM calls that cost the user
latency, and later the curator's share of the work — fed a table that no prompt
ever read. Facts are also the only memory that outlives both the history window
and the in-memory vector store, so nothing at all persisted across restarts.

  before: build_messages(...) -> system prompt with context + emotion only
  after:  build_messages(..., facts=...) -> facts block, newest first, budgeted
"""
from __future__ import annotations

from brain.nodes.generate import build_messages


def fact(key: str, value: str) -> dict:
    return {"key": key, "value": value}


def system_of(messages: list[dict]) -> str:
    return messages[0]["content"]


def test_facts_reach_the_system_prompt():
    """@example: a stored fact -> it appears in the system message the model sees."""
    messages = build_messages(
        "q", "ctx", [], facts=[fact("practice_time", "evening")],
    )

    assert "practice_time: evening" in system_of(messages)


def test_no_facts_leaves_no_empty_heading():
    """@example: an empty facts table -> no dangling "What you know" heading
    burning prompt tokens on nothing."""
    messages = build_messages("q", "ctx", [], facts=[])

    assert "What you know about the user" not in system_of(messages)


def test_facts_block_is_capped_by_its_own_budget():
    """@example: facts totalling more than the budget -> the newest ones fit and
    the rest are cut, because every turn re-sends this block."""
    facts = [fact(f"k{i}", "v" * 100) for i in range(10)]

    messages = build_messages("q", "ctx", [], facts=facts, facts_char_budget=250)

    system = system_of(messages)
    assert "k0" in system
    assert "k1" in system
    assert "k9" not in system


def test_a_single_oversized_fact_is_still_shown():
    """@example: the newest fact alone busts the budget -> keep it rather than
    render an empty block."""
    messages = build_messages(
        "q", "ctx", [], facts=[fact("k", "v" * 5000)], facts_char_budget=100,
    )

    assert "v" * 5000 in system_of(messages)


def test_facts_are_independent_of_the_history_budget():
    """@example: history is trimmed to nothing -> facts still survive, since a
    long conversation must not evict what is known about the user."""
    recent = [{"role": "user", "content": "x" * 9000, "timestamp": "2026-08-09T00:00:00+00:00"}]

    messages = build_messages(
        "q", "ctx", recent,
        history_char_budget=10,
        facts=[fact("goal", "fluency")],
    )

    assert "goal: fluency" in system_of(messages)


def test_facts_precede_the_blocks_that_churn():
    """@example: facts, policy and context in one system prompt -> facts come
    first, because prompt-prefix caching re-prefills everything after the first
    block that changed.

    Measured: the same facts block after the changing policy block costs more
    per turn, before it 0.859s. Reordering is the entire saving — the token
    count is identical."""
    recent = [{
        "role": "assistant", "content": "hi", "timestamp": "2026-08-09T00:00:00+00:00",
    }]

    system = system_of(build_messages(
        "q", "retrieved text", recent, facts=[fact("goal", "fluency")],
        response_policy_instruction="Response style: brief.",
    ))

    assert system.index("goal: fluency") < system.index("Response style: brief.")
    assert system.index("Response style: brief.") < system.index("retrieved text")


def test_facts_and_retrieved_context_both_appear():
    """@example: the two memory sources are separate blocks -> neither replaces
    the other in the prompt."""
    messages = build_messages(
        "q", "a retrieved document", [], facts=[fact("level", "intermediate")],
    )

    system = system_of(messages)
    assert "level: intermediate" in system
    assert "a retrieved document" in system
