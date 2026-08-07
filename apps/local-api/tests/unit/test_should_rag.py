from brain.nodes.should_rag import should_use_rag


def test_short_query_skips_rag():
    assert should_use_rag("ok") is False


def test_conversational_phrase_skips_rag():
    assert should_use_rag("cảm ơn bạn") is False


def test_self_reference_query_skips_rag():
    assert should_use_rag("what did you just say to me") is False


def test_substantive_question_uses_rag():
    assert should_use_rag("how does the vector store handle cosine distance") is True
