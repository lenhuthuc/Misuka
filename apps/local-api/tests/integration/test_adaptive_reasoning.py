import pytest

pytestmark = pytest.mark.asyncio


async def test_chat_enables_thinking_and_expands_total_budget_when_context_is_weak(
    client, fake_brain_bundle,
):
    response = await client.post("/v1/chat", json={
        "query": "Giải thích vì sao bầu trời có màu xanh",
    })

    assert response.status_code == 200
    messages, options = fake_brain_bundle.llm.chat_calls[-1]
    assert fake_brain_bundle.llm.reason_calls[-1][1] == 124
    assert fake_brain_bundle.llm.chat_think_calls[-1] is False
    assert options["num_predict"] == fake_brain_bundle.llm.max_tokens
    assert any("internal analysis and conclusion" in message["content"] for message in messages)


async def test_chat_keeps_thinking_off_when_recent_history_is_relevant(
    client, fake_brain_bundle,
):
    fake_brain_bundle.memory.messages.extend([
        {
            "role": "user",
            "content": "Động cơ điện dùng để làm gì?",
            "timestamp": "2026-08-09T00:00:00+00:00",
        },
        {
            "role": "assistant",
            "content": "Động cơ điện biến điện năng thành chuyển động quay.",
            "timestamp": "2026-08-09T00:00:01+00:00",
        },
    ])

    response = await client.post("/v1/chat", json={
        "query": "Động cơ điện hoạt động như thế nào",
    })

    assert response.status_code == 200
    _messages, options = fake_brain_bundle.llm.chat_calls[-1]
    assert fake_brain_bundle.rag.covered_since_calls == []
    assert fake_brain_bundle.llm.reason_calls == []
    assert fake_brain_bundle.llm.chat_think_calls[-1] is False
    assert options["num_predict"] == fake_brain_bundle.llm.max_tokens
