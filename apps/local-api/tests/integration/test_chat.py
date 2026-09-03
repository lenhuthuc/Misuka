import asyncio


async def test_chat_success_returns_response_and_emotion(client, fake_brain_bundle):
    fake_brain_bundle.llm.response_text = "Chao ban!"

    resp = await client.post("/v1/chat", json={"query": "xin chao"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["response"] == "Chao ban!"
    assert body["emotion"] == "neutral"  # FakeVADService always predicts (0, 0, 0)
    assert set(body["state"].keys()) == {"valence", "arousal", "dominance"}


async def test_chat_uses_rag_for_substantive_queries(client, fake_brain_bundle):
    # Regression for the buffered/streaming policy consolidation
    # (application/conversation_turn.py) — both endpoints must run the same
    # should_use_rag -> rag.build_context -> docs_count pipeline.
    fake_brain_bundle.rag.docs = [{"doc_id": "1", "content": "some fact", "score": 0.9, "metadata": {}}]
    fake_brain_bundle.llm.response_text = "Answer with context"

    resp = await client.post("/v1/chat", json={"query": "how does the vector store handle cosine distance"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["response"] == "Answer with context"
    assert body["docs_count"] == 1


async def test_chat_short_greeting_skips_rag(client, fake_brain_bundle):
    fake_brain_bundle.rag.docs = [{"doc_id": "1", "content": "should not be used", "score": 0.9, "metadata": {}}]

    resp = await client.post("/v1/chat", json={"query": "xin chao"})
    assert resp.status_code == 200
    assert resp.json()["docs_count"] == 0


async def test_chat_weather_query_triggers_web_search_and_reaches_the_prompt(client, fake_brain_bundle):
    # Regression for the buffered/streaming policy consolidation: web search,
    # like RAG, must run through prepare_turn() so both endpoints get it.
    #
    # A prior exchange is seeded because build_messages drops the whole turn
    # note (RAG and web context alike) on a fresh conversation's first turn --
    # see the layout rule in brain/nodes/generate.py.
    await fake_brain_bundle.memory.save_message("user", "chao ban")
    await fake_brain_bundle.memory.save_message("assistant", "chao ban nhe")
    fake_brain_bundle.web_search.results = [
        {"title": "Thời tiết Hà Nội", "body": "Nắng, 32 độ.", "href": "https://example.com"},
    ]
    fake_brain_bundle.web_search.context = "- Thời tiết Hà Nội: Nắng, 32 độ. (nguồn: https://example.com)"

    resp = await client.post("/v1/chat", json={"query": "thời tiết hà nội hôm nay thế nào"})

    assert resp.status_code == 200
    # The engine is given the topic, not the spoken sentence -- the trailing
    # "thế nào" is stripped by should_search_web.to_search_query.
    assert fake_brain_bundle.web_search.query_calls == ["thời tiết hà nội hôm nay"]
    messages, _ = fake_brain_bundle.llm.chat_calls[-1]
    assert any("Nắng, 32 độ." in m["content"] for m in messages)


async def test_chat_conversational_query_skips_web_search(client, fake_brain_bundle):
    # The gate stayed precision-first when the knowledge class was added: chit-chat
    # names no subject the open web could answer, so it still costs no network call.
    fake_brain_bundle.web_search.context = "should not be used"

    resp = await client.post("/v1/chat", json={"query": "hôm nay bạn thấy thế nào"})

    assert resp.status_code == 200
    assert fake_brain_bundle.web_search.query_calls == []


async def test_chat_personal_question_skips_web_search(client, fake_brain_bundle):
    # Phrased like a knowledge question, but only RAG can answer it -- and a
    # search would put a stranger's text in front of a question about the
    # user's own life.
    fake_brain_bundle.web_search.context = "should not be used"

    resp = await client.post("/v1/chat", json={"query": "con mèo nhà mình tên gì"})

    assert resp.status_code == 200
    assert fake_brain_bundle.web_search.query_calls == []


async def test_chat_knowledge_question_is_grounded_before_answering(client, fake_brain_bundle):
    # The third query class. Neither live info nor personal memory, and the
    # one the local weights answer by inventing detail when left ungrounded.
    await fake_brain_bundle.memory.save_message("user", "chao ban")
    await fake_brain_bundle.memory.save_message("assistant", "chao ban nhe")
    fake_brain_bundle.web_search.results = [
        {"title": "Tôm biển", "body": "Sống ở tầng đáy.", "href": "https://example.com"},
    ]
    fake_brain_bundle.web_search.context = "- Tôm biển: Sống ở tầng đáy. (nguồn: https://example.com)"

    resp = await client.post("/v1/chat", json={"query": "kể cho mình nghe về tôm biển nhé"})

    assert resp.status_code == 200
    assert fake_brain_bundle.web_search.query_calls == ["tôm biển"]
    messages, options = fake_brain_bundle.llm.chat_calls[-1]
    assert any("Sống ở tầng đáy." in m["content"] for m in messages)
    # Grounded knowledge is decoded colder and given room for a real answer.
    assert options["temperature"] == 0.30
    assert options["num_predict"] == 480


async def test_chat_knowledge_question_without_results_keeps_chat_decoding(client, fake_brain_bundle):
    # The knowledge treatment is earned by having notes in hand: a longer,
    # colder answer with nothing behind it is a longer, more confident invention.
    await fake_brain_bundle.memory.save_message("user", "chao ban")
    await fake_brain_bundle.memory.save_message("assistant", "chao ban nhe")
    fake_brain_bundle.web_search.results = []
    fake_brain_bundle.web_search.context = ""

    resp = await client.post("/v1/chat", json={"query": "kể cho mình nghe về tôm biển nhé"})

    assert resp.status_code == 200
    _, options = fake_brain_bundle.llm.chat_calls[-1]
    assert options["temperature"] != 0.30
    assert options["num_predict"] != 480


async def test_chat_delegation_inherits_the_topic_it_is_continuing(client, fake_brain_bundle):
    # The observed loop: the assistant offers a choice, the user says "cứ chọn",
    # and the turn that most needs a real fact is the one whose own text names
    # no subject at all. The subject is inherited from the last substantive turn.
    await fake_brain_bundle.memory.save_message("user", "kể cho mình nghe về tôm biển nhé")
    await fake_brain_bundle.memory.save_message("assistant", "bạn muốn nghe về loại nào")
    fake_brain_bundle.web_search.results = [
        {"title": "Tôm biển", "body": "Sống ở tầng đáy.", "href": "https://example.com"},
    ]
    fake_brain_bundle.web_search.context = "- Tôm biển: Sống ở tầng đáy. (nguồn: https://example.com)"

    resp = await client.post("/v1/chat", json={"query": "Ok bạn cứ chọn"})

    assert resp.status_code == 200
    assert fake_brain_bundle.web_search.query_calls == ["tôm biển"]
    messages, _ = fake_brain_bundle.llm.chat_calls[-1]
    note = next(m["content"] for m in messages if m["role"] == "system")
    assert "đừng hỏi lại nữa" in note


async def test_chat_web_search_failure_degrades_to_no_web_context(client, fake_brain_bundle):
    fake_brain_bundle.web_search.error = RuntimeError("network unavailable")

    resp = await client.post("/v1/chat", json={"query": "tin tức mới nhất hôm nay"})

    assert resp.status_code == 200


async def test_chat_filters_a_repeated_assistant_sentence_before_returning_it(client, fake_brain_bundle):
    fake_brain_bundle.llm.response_text = "Câu hỏi cũ bị lặp lại. Nội dung mới được giữ."
    fake_brain_bundle.memory.filter_assistant_response = (
        lambda text, fallback="": text.replace("Câu hỏi cũ bị lặp lại. ", "") or fallback
    )

    resp = await client.post("/v1/chat", json={"query": "hôm nay cũng bình thường"})

    assert resp.status_code == 200
    assert resp.json()["response"] == "Nội dung mới được giữ."


async def test_chat_success_schedules_background_memory_save(client, fake_brain_bundle):
    resp = await client.post("/v1/chat", json={"query": "hello there"})
    assert resp.status_code == 200

    # Background task is fire-and-forget (asyncio.create_task); give the loop
    # a turn to run it before asserting on its side effect.
    await asyncio.sleep(0.05)
    roles = [m["role"] for m in fake_brain_bundle.memory.messages]
    assert roles == ["user", "assistant"]


async def test_chat_vad_controls_policy_without_raw_scores_in_prompt(client, fake_brain_bundle):
    vad = {"valence": -0.7, "arousal": 0.9, "dominance": -0.6}

    resp = await client.post("/v1/chat", json={"query": "toi roi", "user_vad": vad})

    assert resp.status_code == 200
    body = resp.json()
    assert body["response_policy"]["max_tokens"] == 256
    assert body["response_policy"]["stream_pace"] == "immediate"
    messages, options = fake_brain_bundle.llm.chat_calls[-1]
    assert options == {"temperature": 0.55, "num_predict": 256}
    # A fresh conversation cannot carry a leading system note: that would
    # suppress the persona from the model's Modelfile. VAD still controls the
    # hard generation options on this first turn.
    assert messages == [{"role": "user", "content": "toi roi"}]
    assert "-0.7" not in messages[0]["content"]


async def test_seed_endpoint_inserts_docs(client, fake_brain_bundle):
    resp = await client.post("/v1/chat/seed")
    assert resp.status_code == 200
    assert resp.json()["inserted"] == len(fake_brain_bundle.vector.upserted[0][0])


async def test_chat_regenerates_once_when_the_whole_reply_is_a_repetition(client, fake_brain_bundle):
    """The case BM25 exists for. Emitting the raw text here handed the user
    back the very sentence the filter had just caught."""
    fake_brain_bundle.llm.response_text = "Bạn muốn nghe chuyện khác hay tìm chủ đề mới?"
    fake_brain_bundle.memory.messages.extend([
        {"role": "user", "content": "kể chuyện gì đi", "vad": None, "emotion": None,
         "timestamp": "2026-09-02T13:00:00+00:00"},
        {"role": "assistant", "content": "Bạn muốn nghe chuyện khác hay tìm chủ đề mới?",
         "vad": None, "emotion": None, "timestamp": "2026-09-02T13:00:01+00:00"},
    ])

    calls: list[list[dict]] = []

    async def chat(messages, options=None, *, think: bool = False):
        calls.append(messages)
        # First pass repeats; the retry, having been told so, writes something new.
        return "Câu trả lời mới, không lặp." if len(calls) > 1 else fake_brain_bundle.llm.response_text

    fake_brain_bundle.llm.chat = chat
    fake_brain_bundle.memory.filter_assistant_response = (
        lambda text, fallback="": "" if "chủ đề mới" in text else text
    )

    resp = await client.post("/v1/chat", json={"query": "chủ đề mới để kể"})

    assert resp.status_code == 200
    assert resp.json()["response"] == "Câu trả lời mới, không lặp."
    assert len(calls) == 2
    # The retry has to say *why*, and land where the note is read -- never at
    # index 0, which would silently replace the Modelfile persona.
    assert calls[1][0]["role"] != "system"
    assert "lặp lại" in calls[1][-2]["content"]
    assert calls[1][-1]["role"] == "user"


async def test_chat_keeps_the_raw_reply_when_the_retry_repeats_too(client, fake_brain_bundle):
    """Last resort stays the model's own words: a fixed canned phrase would
    join the BM25 corpus and start colliding with future turns."""
    repeated = "Bạn muốn nghe chuyện khác hay tìm chủ đề mới?"
    fake_brain_bundle.llm.response_text = repeated
    fake_brain_bundle.memory.filter_assistant_response = lambda text, fallback="": ""

    resp = await client.post("/v1/chat", json={"query": "chủ đề mới để kể"})

    assert resp.status_code == 200
    assert resp.json()["response"] == repeated


async def test_chat_does_not_regenerate_when_some_sentences_survive(client, fake_brain_bundle):
    calls: list[list[dict]] = []

    async def chat(messages, options=None, *, think: bool = False):
        calls.append(messages)
        return "Câu cũ bị lặp. Câu mới được giữ."

    fake_brain_bundle.llm.chat = chat
    fake_brain_bundle.memory.filter_assistant_response = (
        lambda text, fallback="": text.replace("Câu cũ bị lặp. ", "") or fallback
    )

    resp = await client.post("/v1/chat", json={"query": "hôm nay bình thường"})

    assert resp.json()["response"] == "Câu mới được giữ."
    assert len(calls) == 1
