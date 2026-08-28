"""The deterministic half of the no-closing-question guard.

The prompt half (`_DELIVER_NOW_INSTRUCTION`) reached two clean replies in three
against mitsuka-ft. These cover the third.
"""
from brain.trailing_question import (
    TrailingQuestionSuppressor,
    is_question,
    strip_trailing_question,
)


def test_a_trailing_question_is_removed():
    text = "Chọn tôm hùm nhé. Nó sống ở vùng rạn đá. Bạn muốn nghe về cách chúng sinh sản?"

    assert strip_trailing_question(text) == "Chọn tôm hùm nhé. Nó sống ở vùng rạn đá."


def test_a_question_in_the_middle_of_a_reply_is_left_alone():
    # The rule is "do not hand the turn back", not "never use a question mark".
    text = "Bạn hỏi tôm hùm à? Nó sống ở vùng rạn đá."

    assert strip_trailing_question(text) == text


def test_an_unpunctuated_vietnamese_question_still_counts():
    # Vietnamese closes a yes/no question on a final particle, and the model
    # does sometimes end one with a full stop.
    assert is_question("Bạn thấy thú vị không.") is True
    # The same word mid-sentence is a negation, which is far more common.
    assert is_question("Tôm hùm không sống ở nước ngọt.") is False


def test_a_reply_that_is_nothing_but_a_question_is_kept():
    # Same failure mode the repetition filter chose: a reply that breaks the
    # guard beats no reply at all.
    text = "Bạn muốn nghe về loại nào?"

    assert strip_trailing_question(text) == text


def test_the_stream_suppressor_drops_a_question_that_ends_the_reply():
    s = TrailingQuestionSuppressor()

    assert s.feed("Chọn tôm hùm nhé. ") == ["Chọn tôm hùm nhé. "]
    assert s.feed("Bạn muốn nghe thêm?") == []      # held, not yet emitted
    assert s.flush() == []                          # end of stream: dropped


def test_the_stream_suppressor_releases_a_question_that_turns_out_to_be_mid_reply():
    s = TrailingQuestionSuppressor()

    assert s.feed("Bạn hỏi tôm hùm à?") == []
    assert s.feed("Nó sống ở rạn đá.") == ["Bạn hỏi tôm hùm à?", "Nó sống ở rạn đá."]
    assert s.flush() == []


def test_the_stream_suppressor_keeps_a_reply_that_is_only_a_question():
    s = TrailingQuestionSuppressor()

    assert s.feed("Bạn muốn nghe về loại nào?") == []
    assert s.flush() == ["Bạn muốn nghe về loại nào?"]
