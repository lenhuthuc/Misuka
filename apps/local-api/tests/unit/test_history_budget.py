"""Tests for bounding the verbatim history window by characters, and for the
message layout that keeps the fine-tune's persona alive.

ROOT CAUSE the budget tests pin down:

`memory_recent_limit` caps how many messages the prompt carries, but not how
large they are. Once retrieved-context duplication was fixed, the prompt still
climbed 7,834 -> 11,134 characters across five turns purely from the history
window, because the model was writing 2,000-2,900 character replies and every
one of them was re-sent on every later turn. Prefill is linear in prompt
length, so each turn paid for all the verbosity before it.

  before: build_messages(...) -> all `recent` rows, whatever their size
  after:  build_messages(..., history_char_budget=N) -> newest rows that fit N

ROOT CAUSE the layout tests pin down:

The persona moved out of this codebase and into the `mitsuka-ft` fine-tune's
Modelfile. Ollama injects that SYSTEM block only when the caller's first
message is not itself a system message, so a system message at index 0 -- which
is exactly where the old prompt sat -- silently replaces the persona instead of
adding to it. Asked "bạn tên gì thế?" with one at index 0, the model dropped
its name and its pronoun: "Tên tôi là Bố". Nothing raises; the only guard is
the layout, so the layout is tested.
"""
from __future__ import annotations

from brain.nodes.generate import build_messages


def msg(role: str, content: str, emotion: str | None = None) -> dict:
    row = {"role": role, "content": content, "timestamp": "2026-08-09T02:00:00+00:00"}
    if emotion is not None:
        row["emotion"] = emotion
    return row


def history_of(messages: list[dict]) -> list[dict]:
    """The replayed conversation: everything but the turn note and the query."""
    return [m for m in messages if m["role"] != "system"][:-1]


def test_oldest_messages_are_dropped_first():
    """@example: three 100-char exchanges against a 250-char budget -> the two
    newest complete exchanges survive and the oldest pair is dropped."""
    recent = [
        msg("user", "a" * 50), msg("assistant", "b" * 50),
        msg("user", "c" * 50), msg("assistant", "d" * 50),
        msg("user", "e" * 50), msg("assistant", "f" * 50),
    ]

    messages = build_messages("now what?", "ctx", recent, history_char_budget=250)

    assert [m["content"][0] for m in history_of(messages)] == ["c", "d", "e", "f"]


def test_history_keeps_chronological_order_after_trimming():
    """@example: trimming drops from the front -> what remains is still oldest
    to newest, not reversed."""
    recent = [
        msg("user", "one"), msg("assistant", "two"),
        msg("user", "three"), msg("assistant", "four"),
    ]

    messages = build_messages("q", "ctx", recent, history_char_budget=10_000)

    assert [m["content"] for m in history_of(messages)] == ["one", "two", "three", "four"]


def test_a_single_oversized_exchange_is_still_kept_as_a_pair():
    """@example: the newest exchange alone busts the budget -> retain both its
    question and answer rather than splitting the pair."""
    recent = [msg("user", "old"), msg("assistant", "z" * 9000)]

    messages = build_messages("q", "ctx", recent, history_char_budget=1000)

    history = history_of(messages)
    assert len(history) == 2
    assert [m["role"] for m in history] == ["user", "assistant"]
    assert history[1]["content"] == "z" * 9000


def test_stored_assistant_emotion_is_not_reinjected_as_a_fixed_prompt():
    """@example: emotion labels remain output metadata, not a stale instruction
    that can override the current user's emotional state on later turns."""
    recent = [
        msg("user", "c" * 100),
        msg("assistant", "d" * 100, emotion="joy"),
        msg("user", "e" * 1500),
        msg("assistant", "f" * 1500),
    ]

    messages = build_messages("q", "ctx", recent, history_char_budget=3000)

    assert not any("joy" in m["content"] for m in messages)
    assert len(history_of(messages)) < len(recent)


def test_orphan_messages_at_history_boundaries_are_dropped():
    """A LIMIT boundary may begin on an assistant, and a partial write may end
    on a user; neither orphan is sent to Ollama."""
    recent = [
        msg("assistant", "orphan answer"),
        msg("user", "paired question"),
        msg("assistant", "paired answer"),
        msg("user", "orphan question"),
    ]

    messages = build_messages("new query", "", recent, history_char_budget=10_000)

    assert history_of(messages) == [
        {"role": "user", "content": "paired question"},
        {"role": "assistant", "content": "paired answer"},
    ]


def test_query_is_always_the_final_message():
    """@example: whatever trimming does to history -> the user's actual question
    stays last, where the model expects it."""
    recent = [msg("user", "g" * 5000)]

    messages = build_messages("the real question", "ctx", recent, history_char_budget=100)

    assert messages[-1] == {"role": "user", "content": "the real question"}


def test_first_message_is_never_a_system_message():
    """@example: a full turn -- history, retrieved context and an active policy
    -> index 0 is still a replayed conversation turn, which is the only thing
    that makes Ollama inject the Modelfile persona."""
    recent = [msg("user", "chào bạn"), msg("assistant", "Chào bạn, hôm nay thế nào?")]

    messages = build_messages(
        "bạn tên gì thế?", "người dùng nuôi một con mèo tên Miu", recent,
        response_policy_instruction="Response style: brief; calm.",
    )

    assert messages[0]["role"] != "system"


def test_turn_note_sits_directly_before_the_query():
    """@example: the note is the last thing the model reads before the question
    -- late enough not to invalidate the cached history prefix in front of it,
    and never mistaken for the user's own words."""
    recent = [msg("user", "chào bạn"), msg("assistant", "Chào bạn.")]

    messages = build_messages("bạn tên gì thế?", "mèo tên Miu", recent)

    assert messages[-2]["role"] == "system"
    assert "Miu" in messages[-2]["content"]
    assert messages[-1] == {"role": "user", "content": "bạn tên gì thế?"}


def test_history_turn_adds_continuity_guard_even_without_rag():
    """@example: existing conversation -> add a late continuity guard so the
    model does not imitate repetitive questions from its own prior replies."""
    recent = [msg("user", "chào bạn"), msg("assistant", "Chào bạn.")]

    messages = build_messages("khỏe không", "", recent, response_policy_instruction="")

    assert [m["role"] for m in messages] == ["user", "assistant", "system", "user"]
    assert "không lặp lại câu hỏi" in messages[-2]["content"]
    assert "không cần kết thúc" in messages[-2]["content"]


def test_declining_to_elaborate_gets_a_strict_no_followup_policy():
    recent = [
        msg("user", "hôm nay bình thường"),
        msg("assistant", "Bạn kể thêm cho mình được không?"),
    ]

    messages = build_messages(
        "Tôi đã bảo là hôm nay tôi không thấy có cái gì đặc biệt hết mà",
        "",
        recent,
    )

    assert "Tuyệt đối không hỏi" in messages[-2]["content"]
    assert "không mời người dùng kể/chia sẻ thêm" in messages[-2]["content"]


def test_empty_context_turn_gets_a_no_hollow_promise_guard():
    """@example: no RAG/web context this turn -> tell the model not to promise
    content ("mình sẽ kể chủ đề đó cho bạn") it has nothing behind, since that
    is exactly the shape of the empty-context deflection this guard exists to
    stop."""
    recent = [msg("user", "chào bạn"), msg("assistant", "Chào bạn.")]

    messages = build_messages("kể cho mình nghe đi", "", recent)

    assert "Đừng hứa sẽ kể" in messages[-2]["content"]


def test_nonempty_context_turn_skips_the_no_hollow_promise_guard():
    """@example: real retrieved context this turn -> the guard is redundant
    (there is real material to draw from) and would only cost prefill, so it
    is not added alongside the context block."""
    recent = [msg("user", "chào bạn"), msg("assistant", "Chào bạn.")]

    messages = build_messages("kể cho mình nghe đi", "mèo tên Miu", recent)

    assert "Đừng hứa sẽ kể" not in messages[-2]["content"]


def test_empty_history_drops_the_note_rather_than_leading_with_it():
    """@example: first turn of a fresh conversation -> the note has no position
    that is not index 0, so it is dropped; losing the persona costs more than
    losing one turn of background notes."""
    messages = build_messages("hello", "ctx", [], history_char_budget=3000)

    assert messages == [{"role": "user", "content": "hello"}]
