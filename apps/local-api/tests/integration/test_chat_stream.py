"""SSE contract for /v1/chat/stream — versioned, discriminated envelope.

See schemas/chat.py: every event carries `type` and `turn_id`. The [DONE]
sentinel string (baseline pre-Phase-2 behavior) is gone — `done` is now a
JSON event like everything else.
"""
import json


def _parse_sse(raw: str) -> list[dict]:
    events: list[dict] = []
    for block in raw.split("\n\n"):
        block = block.strip()
        if not block or block.startswith(":"):
            continue
        for line in block.splitlines():
            if line.startswith("data:"):
                events.append(json.loads(line[len("data:"):].strip()))
    return events


async def test_chat_stream_success_yields_deltas_emotion_then_done(client, fake_brain_bundle):
    fake_brain_bundle.llm.stream_chunks = ["Xin ", "chao"]

    resp = await client.post("/v1/chat/stream", json={"query": "hello"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")

    events = _parse_sse(resp.text)

    # Every event shares one turn_id.
    turn_ids = {e["turn_id"] for e in events}
    assert len(turn_ids) == 1

    deltas = [e for e in events if e["type"] == "delta"]
    assert [e["content"] for e in deltas] == ["Xin ", "chao"]

    emotion_events = [e for e in events if e["type"] == "emotion"]
    assert len(emotion_events) == 1
    assert emotion_events[0]["emotion"] == "neutral"
    assert set(emotion_events[0]["state"].keys()) == {"valence", "arousal", "dominance"}

    assert events[-1]["type"] == "done"


async def test_chat_stream_llm_failure_mid_stream_yields_typed_error_event(client, fake_brain_bundle):
    fake_brain_bundle.llm.stream_error = RuntimeError("ollama unreachable")

    resp = await client.post("/v1/chat/stream", json={"query": "hello"})

    # HTTP status stays 200 — the failure is an in-band SSE event so the
    # browser (which already committed to reading a stream) can observe it.
    assert resp.status_code == 200
    events = _parse_sse(resp.text)

    error_events = [e for e in events if e["type"] == "error"]
    assert len(error_events) == 1
    assert error_events[0]["error"]["code"] == "LLM_UNAVAILABLE"
    assert error_events[0]["error"]["retryable"] is True
    assert "ollama unreachable" in error_events[0]["error"]["message"]

    assert events[-1]["type"] == "done"


async def test_chat_stream_no_response_skips_emotion_event(client, fake_brain_bundle):
    fake_brain_bundle.llm.stream_error = RuntimeError("down")

    resp = await client.post("/v1/chat/stream", json={"query": "hello"})
    events = _parse_sse(resp.text)

    assert [e for e in events if e["type"] == "emotion"] == []  # empty full_response -> skipped entirely


async def test_chat_stream_vad_uses_fast_short_generation_policy(client, fake_brain_bundle):
    resp = await client.post("/v1/chat/stream", json={
        "query": "toi dang roi", "user_vad": {"valence": -0.7, "arousal": 0.9, "dominance": -0.6},
    })

    assert resp.status_code == 200
    messages, options = fake_brain_bundle.llm.stream_chat_calls[-1]
    assert options == {"temperature": 0.55, "num_predict": 256}
    assert "Response style: brief" in messages[0]["content"]
