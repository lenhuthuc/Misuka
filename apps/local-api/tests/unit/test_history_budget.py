"""Tests for bounding the verbatim history window by characters.

ROOT CAUSE these tests pin down:

`memory_recent_limit` caps how many messages the prompt carries, but not how
large they are. Once retrieved-context duplication was fixed, the prompt still
climbed 7,834 -> 11,134 characters across five turns purely from the history
window, because the model was writing 2,000-2,900 character replies and every
one of them was re-sent on every later turn. Prefill is linear in prompt
length, so each turn paid for all the verbosity before it.

  before: build_messages(...) -> all `recent` rows, whatever their size
  after:  build_messages(..., history_char_budget=N) -> newest rows that fit N
"""
from __future__ import annotations

from brain.nodes.generate import build_messages


def msg(role: str, content: str, emotion: str | None = None) -> dict:
    row = {"role": role, "content": content, "timestamp": "2026-08-09T02:00:00+00:00"}
    if emotion is not None:
        row["emotion"] = emotion
    return row


def test_oldest_messages_are_dropped_first():
    """@example: three 100-char messages against a 250-char budget -> the two
    newest survive and the oldest is dropped."""
    recent = [msg("user", "a" * 100), msg("assistant", "b" * 100), msg("user", "c" * 100)]

    messages = build_messages("now what?", "ctx", recent, history_char_budget=250)

    history = messages[1:-1]
    assert [m["content"][0] for m in history] == ["b", "c"]


def test_history_keeps_chronological_order_after_trimming():
    """@example: trimming drops from the front -> what remains is still oldest
    to newest, not reversed."""
    recent = [msg("user", "one"), msg("assistant", "two"), msg("user", "three")]

    messages = build_messages("q", "ctx", recent, history_char_budget=10_000)

    assert [m["content"] for m in messages[1:-1]] == ["one", "two", "three"]


def test_a_single_oversized_message_is_still_kept():
    """@example: the newest message alone busts the budget -> keep it, so the
    model never loses the turn it is answering."""
    recent = [msg("user", "old"), msg("assistant", "z" * 9000)]

    messages = build_messages("q", "ctx", recent, history_char_budget=1000)

    history = messages[1:-1]
    assert len(history) == 1
    assert history[0]["content"] == "z" * 9000


def test_emotion_is_read_from_the_untrimmed_window():
    """@example: the only emotion-bearing message is old enough to be trimmed
    from the prompt -> the agent's carried-over mood still reaches the system
    prompt, because trimming is a latency measure and not a memory reset."""
    recent = [
        msg("assistant", "d" * 100, emotion="joy"),
        msg("user", "e" * 3000),
        msg("assistant", "f" * 3000),
    ]

    messages = build_messages("q", "ctx", recent, history_char_budget=3000)

    assert "joy" in messages[0]["content"]
    assert len(messages[1:-1]) < len(recent)


def test_query_is_always_the_final_message():
    """@example: whatever trimming does to history -> the user's actual question
    stays last, where the model expects it."""
    recent = [msg("user", "g" * 5000)]

    messages = build_messages("the real question", "ctx", recent, history_char_budget=100)

    assert messages[-1] == {"role": "user", "content": "the real question"}


def test_empty_history_produces_system_plus_query_only():
    """@example: first turn of a fresh conversation -> exactly two messages."""
    messages = build_messages("hello", "ctx", [], history_char_budget=3000)

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["content"] == "hello"
