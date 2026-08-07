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


async def test_chat_success_schedules_background_memory_save(client, fake_brain_bundle):
    resp = await client.post("/v1/chat", json={"query": "hello there"})
    assert resp.status_code == 200

    # Background task is fire-and-forget (asyncio.create_task); give the loop
    # a turn to run it before asserting on its side effect.
    await asyncio.sleep(0.05)
    roles = [m["role"] for m in fake_brain_bundle.memory.messages]
    assert roles == ["user", "assistant"]


async def test_seed_endpoint_inserts_docs(client, fake_brain_bundle):
    resp = await client.post("/v1/chat/seed")
    assert resp.status_code == 200
    assert resp.json()["inserted"] == len(fake_brain_bundle.vector.upserted[0][0])
