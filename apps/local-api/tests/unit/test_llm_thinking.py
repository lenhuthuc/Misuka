import json

import httpx
import pytest

from brain.llm_service import LLMService


@pytest.mark.asyncio
async def test_reason_uses_dedicated_model_and_a_hard_token_cap():
    captured: dict = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={
            "message": {"content": "Phân tích nội bộ."},
        })

    llm = LLMService(
        "http://ollama.test", "mitsuka-ft", max_tokens=320,
        reasoning_model="qwen3:1.7b",
    )
    await llm._client.aclose()
    llm._client = httpx.AsyncClient(
        base_url="http://ollama.test",
        transport=httpx.MockTransport(handler),
    )

    try:
        response = await llm.reason(
            [{"role": "user", "content": "Một câu hỏi khó"}],
            max_tokens=192,
        )
    finally:
        await llm.aclose()

    assert response == "Phân tích nội bộ."
    assert captured["model"] == "qwen3:1.7b"
    assert captured["think"] is False
    assert captured["options"]["num_predict"] == 192


def test_hidden_reasoning_note_keeps_a_fresh_turn_as_one_user_message():
    from brain.reasoning_policy import inject_reasoning_note

    messages = inject_reasoning_note(
        [{"role": "user", "content": "Câu hỏi"}],
        "Phân tích",
    )

    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert "Phân tích" in messages[0]["content"]
    assert messages[0]["content"].endswith("Lời người dùng hiện tại:\nCâu hỏi")


def test_hidden_reasoning_note_uses_a_late_system_message_when_history_exists():
    from brain.reasoning_policy import inject_reasoning_note

    original = [
        {"role": "user", "content": "Chuyện cũ"},
        {"role": "assistant", "content": "Mình nhớ."},
        {"role": "user", "content": "Câu hỏi mới"},
    ]
    messages = inject_reasoning_note(original, "Phân tích")

    assert messages[0] == original[0]
    assert messages[-2]["role"] == "system"
    assert messages[-1] == original[-1]
