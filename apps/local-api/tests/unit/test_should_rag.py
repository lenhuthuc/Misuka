from brain.nodes.should_rag import is_declining_to_elaborate, is_delegating_choice, should_use_rag


def test_short_query_skips_rag():
    assert should_use_rag("ok") is False


def test_conversational_phrase_skips_rag():
    assert should_use_rag("cảm ơn bạn") is False


def test_self_reference_query_skips_rag():
    assert should_use_rag("what did you just say to me") is False


def test_vietnamese_correction_of_prior_turn_skips_rag():
    assert should_use_rag("Tôi đã bảo là hôm nay không có gì đặc biệt hết mà") is False
    assert should_use_rag("Mình đã nói rồi mà") is False


def test_declining_to_elaborate_is_narrower_than_a_generic_correction():
    assert is_declining_to_elaborate("Hôm nay không có gì đặc biệt") is True
    assert is_declining_to_elaborate("Tôi đã bảo là tôi muốn đi Nhật mà") is False


def test_substantive_question_uses_rag():
    assert should_use_rag("how does the vector store handle cosine distance") is True


def test_delegating_the_choice_back_is_detected():
    # The turn the loop-breaking guard keys on: the user has answered the
    # assistant's question by handing the decision back.
    assert is_delegating_choice("Ok bạn cứ chọn") is True
    assert is_delegating_choice("tùy bạn thôi") is True
    assert is_delegating_choice("bạn kể đi") is True
    assert is_delegating_choice("sao cũng được") is True
    assert is_delegating_choice("you decide") is True


def test_the_adverb_between_pronoun_and_verb_is_open_ended():
    """Regression: the slot used to accept only "cứ" and "tự", so "bạn tự động
    chọn" was not delegation -- and that one miss switched off the deliver-now
    instruction, the trailing-question suppressor and RAG suppression together,
    which is what let the assistant loop back to offering the same choice."""
    for phrasing in (
        "bạn tự động chọn",
        "bạn tự động chọn giúp mình",
        "bạn cứ tự nhiên chọn",
        "bạn tuỳ ý chọn",
        "bạn chủ động quyết định nhé",
    ):
        assert is_delegating_choice(phrasing) is True, phrasing


def test_both_spellings_of_tuy_are_delegation():
    """"tùy" and "tuỳ" are the same word with the tone mark on a different
    vowel, so one character class cannot cover both."""
    assert is_delegating_choice("tùy bạn") is True
    assert is_delegating_choice("tuỳ bạn") is True


def test_the_choice_handed_back_as_a_condition():
    assert is_delegating_choice("bạn thích cái nào thì chọn") is True


def test_delegation_does_not_swallow_ordinary_questions_or_self_reference():
    # "bạn nói" heads a question far more often than a delegation, so the
    # speech verbs are only admitted with a following imperative particle.
    assert is_delegating_choice("bạn nói gì thế") is False
    # The bare form is only delegation at the head of the utterance.
    assert is_delegating_choice("mình cứ nói thôi") is False
    # An interrogative straight after the verb makes it the user asking, not
    # the user deferring -- including when the verb itself is two words, where
    # the pattern must not fall back to matching just "quyết".
    assert is_delegating_choice("bạn chọn cái gì vậy") is False
    assert is_delegating_choice("bạn quyết định gì rồi") is False
    assert is_delegating_choice("bạn chọn ai làm đội trưởng") is False


def test_delegation_skips_rag():
    # "ok bạn cứ chọn" names no subject: its embedding ranks memories by tone,
    # not topic, so retrieval can only return a best-of-a-bad-lot hit.
    assert should_use_rag("Ok bạn cứ chọn") is False


def test_the_imperative_particle_may_trail_the_verb():
    """Regression: "bạn cứ nói về nó đi" is the same delegation as "bạn nói đi",
    and it appeared three times in one observed loop without ever being
    detected -- the particle just was not adjacent to the verb."""
    for phrasing in (
        "bạn cứ nói về nó đi",
        "bạn nói về chủ đề đó đi",
        "bạn kể chuyện đó đi",
        "bạn cụ thể cho mình về chủ đề đó đi",
    ):
        assert is_delegating_choice(phrasing) is True, phrasing


def test_an_interrogative_between_verb_and_particle_is_still_a_question():
    assert is_delegating_choice("bạn nói câu đó nghĩa là gì") is False
    assert is_delegating_choice("bạn cụ thể là muốn gì") is False
    assert is_delegating_choice("bạn kể chuyện gì thế") is False
