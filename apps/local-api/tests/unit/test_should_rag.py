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


def test_delegation_does_not_swallow_ordinary_questions_or_self_reference():
    # "bạn nói" heads a question far more often than a delegation, so the
    # speech verbs are only admitted with a following imperative particle.
    assert is_delegating_choice("bạn nói gì thế") is False
    # The bare form is only delegation at the head of the utterance.
    assert is_delegating_choice("mình cứ nói thôi") is False


def test_delegation_skips_rag():
    # "ok bạn cứ chọn" names no subject: its embedding ranks memories by tone,
    # not topic, so retrieval can only return a best-of-a-bad-lot hit.
    assert should_use_rag("Ok bạn cứ chọn") is False
