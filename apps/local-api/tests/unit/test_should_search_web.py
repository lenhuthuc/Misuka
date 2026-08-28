from brain.nodes.should_search_web import (
    decide_web_search,
    is_knowledge_question,
    should_use_web_search,
    to_search_query,
)


def test_weather_query_triggers_web_search():
    assert should_use_web_search("thời tiết Hà Nội hôm nay thế nào") is True


def test_news_query_triggers_web_search():
    assert should_use_web_search("tin tức mới nhất về AI") is True


def test_price_query_triggers_web_search():
    assert should_use_web_search("giá vàng hôm nay bao nhiêu") is True


def test_english_weather_query_triggers_web_search():
    assert should_use_web_search("what is the weather forecast today") is True


def test_product_recommendation_query_triggers_web_search():
    assert should_use_web_search("có sản phẩm nào tốt trên thị trường") is True


def test_which_is_better_comparison_triggers_web_search():
    assert should_use_web_search("nên mua điện thoại hãng nào") is True
    assert should_use_web_search("so sánh iPhone 16 với Samsung S25") is True


def test_personal_memory_question_does_not_trigger_web_search():
    # RAG's job, not web search's -- nothing on the open web knows this.
    assert should_use_web_search("con mèo nhà mình tên gì") is False


def test_bare_time_marker_without_a_topic_does_not_trigger_web_search():
    # "hôm nay" alone names no topic to search for.
    assert should_use_web_search("hôm nay bạn thấy thế nào") is False


def test_conversational_greeting_does_not_trigger_web_search():
    assert should_use_web_search("xin chào") is False


# ── The knowledge class ──────────────────────────────────────────────────────
# Neither live nor personal: an ordinary open-world question. Left ungrounded
# these are what the local weights answer by inventing detail.

def test_open_world_question_is_a_knowledge_question():
    assert is_knowledge_question("tôm biển là con gì") is True
    assert is_knowledge_question("kể cho mình nghe về tôm biển nhé") is True
    assert is_knowledge_question("giải thích tại sao trời mưa") is True
    assert is_knowledge_question("what is a mantis shrimp") is True


def test_personal_scope_outranks_knowledge_phrasing():
    # Phrased like a knowledge question, answerable only from RAG.
    assert is_knowledge_question("con mèo nhà mình tên gì") is False
    assert is_knowledge_question("bạn tên gì thế") is False
    assert is_knowledge_question("mình là ai trong mắt bạn") is False


def test_chit_chat_is_not_a_knowledge_question():
    assert is_knowledge_question("hôm nay bạn thấy thế nào") is False
    assert is_knowledge_question("xin chào") is False


def test_knowledge_class_can_be_switched_off_without_touching_the_live_gate():
    assert decide_web_search("tôm biển là con gì", knowledge_enabled=False).kind == ""
    assert decide_web_search("giá vàng hôm nay bao nhiêu", knowledge_enabled=False).kind == "live"


# ── Search-query normalisation ───────────────────────────────────────────────

def test_the_engine_is_given_the_topic_not_the_spoken_sentence():
    assert to_search_query("kể cho mình nghe về tôm biển nhé") == "tôm biển"
    assert to_search_query("giải thích tại sao trời mưa") == "tại sao trời mưa"
    assert to_search_query("tell me about deep sea shrimp?") == "deep sea shrimp"


# ── Topic carry-over ─────────────────────────────────────────────────────────

def test_delegation_inherits_the_subject_from_the_last_substantive_turn():
    # The turn that most needs grounding is the one with no topic in it.
    recent = [
        {"role": "user", "content": "kể cho mình nghe về tôm biển nhé"},
        {"role": "assistant", "content": "bạn muốn nghe về loại nào"},
    ]
    decision = decide_web_search("Ok bạn cứ chọn", recent)

    assert decision.kind == "knowledge"
    assert decision.query == "tôm biển"
    assert decision.inherited is True


def test_delegation_after_a_chat_turn_inherits_nothing():
    recent = [{"role": "user", "content": "hôm nay mình mệt quá"}]

    assert decide_web_search("Ok bạn cứ chọn", recent).should_search is False


def test_only_the_newest_substantive_turn_is_consulted():
    # Scanning further back would resurrect an abandoned subject.
    recent = [
        {"role": "user", "content": "kể cho mình nghe về tôm biển nhé"},
        {"role": "assistant", "content": "..."},
        {"role": "user", "content": "thôi mình đói rồi"},
    ]

    assert decide_web_search("tùy bạn", recent).should_search is False


def test_a_query_with_its_own_topic_never_inherits():
    recent = [{"role": "user", "content": "kể cho mình nghe về tôm biển nhé"}]
    decision = decide_web_search("giá vàng hôm nay bao nhiêu", recent)

    assert decision.kind == "live"
    assert decision.inherited is False
