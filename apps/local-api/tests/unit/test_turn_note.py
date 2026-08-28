"""The per-turn system note built by `brain.nodes.generate`.

These cover the loop this note was rewritten to break. The assistant offered a
choice, the user answered it with "ok bạn cứ chọn", and the assistant offered
the choice again -- because the note it was reading ended with an explicit
"hoặc hỏi ngược người dùng muốn nghe về gì", and a 1.7B weights what it read
last. The fix is two-sided: withhold that clause once the user has answered it,
and put the block that forbids a question where nothing can be read after it.
"""
from brain.nodes.generate import build_messages


def _note(messages: list[dict[str, str]]) -> str:
    return next((m["content"] for m in messages if m["role"] == "system"), "")


# A prior exchange is required in every case: `build_messages` drops the note
# entirely on a fresh conversation's first turn, because there is no position
# for it that is not index 0 -- see the layout rule in nodes/generate.py.
_HISTORY = [
    {"role": "user", "content": "kể cho mình nghe về tôm biển nhé"},
    {"role": "assistant", "content": "bạn muốn nghe về loại nào"},
]


def test_delegation_forbids_asking_the_question_again():
    note = _note(build_messages("Ok bạn cứ chọn", "", _HISTORY))

    assert "đừng hỏi lại nữa" in note
    # The escape hatch the model took last time is gone from this turn.
    assert "hỏi ngược" not in note


def test_the_ask_back_clause_survives_when_the_user_has_not_answered_it():
    # Still the right out on an ordinary empty-context turn: the assistant has
    # nothing and the user has not said what they want.
    note = _note(build_messages("kể mình nghe chuyện gì đó", "", _HISTORY))

    assert "hỏi ngược" in note
    assert "đừng hỏi lại nữa" not in note


def test_the_no_question_block_is_read_last():
    note = _note(build_messages("Ok bạn cứ chọn", "ghi chú nền", _HISTORY))

    # Ordering is load-bearing, not cosmetic: every other block can be
    # satisfied by a question, and this is the only one that forbids one.
    assert note.rstrip().endswith("nói thẳng là mình không chắc.")


def test_grounded_knowledge_overrides_the_fine_tunes_sentence_budget():
    note = _note(build_messages("tôm biển là con gì", "ghi chú nền", _HISTORY, grounded_knowledge=True))

    assert "3–5 câu" in note
    assert "không kết thúc bằng câu hỏi" in note


def test_an_ungrounded_turn_gets_no_extra_room_to_talk():
    # Without notes in front of it, more room to talk is more room to invent.
    note = _note(build_messages("tôm biển là con gì", "", _HISTORY))

    assert "3–5 câu" not in note
