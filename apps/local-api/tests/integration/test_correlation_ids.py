"""Request/turn correlation end-to-end through the real ASGI stack.

A turn should be traceable via `turn_id` from the HTTP response all the way
to the background memory task's own log lines (see core/logging.py's
`bind_turn_id` docstring for why this works through a spawned asyncio.Task).
"""
import json
import logging


async def test_every_response_carries_a_request_id_header(client):
    resp = await client.get("/health/live")
    assert resp.headers["X-Request-Id"]


async def test_request_id_is_echoed_back_when_client_supplies_one(client):
    resp = await client.get("/health/live", headers={"X-Request-Id": "client-supplied-id"})
    assert resp.headers["X-Request-Id"] == "client-supplied-id"


async def test_chat_response_includes_turn_id(client):
    resp = await client.post("/v1/chat", json={"query": "xin chao"})
    assert resp.status_code == 200
    assert resp.json()["turn_id"]


async def test_chat_stream_turn_id_header_matches_sse_event_turn_ids(client):
    resp = await client.post("/v1/chat/stream", json={"query": "hello"})
    header_turn_id = resp.headers["X-Turn-Id"]

    event_turn_ids = {
        json.loads(line[len("data:"):].strip())["turn_id"]
        for line in resp.text.splitlines()
        if line.startswith("data:") and "[DONE]" not in line
    }
    assert event_turn_ids == {header_turn_id}


async def test_background_memory_task_log_lines_carry_the_turn_id(client, fake_brain_bundle, caplog):
    with caplog.at_level(logging.INFO, logger="brain.background"):
        resp = await client.post("/v1/chat", json={"query": "hello there"})
        turn_id = resp.json()["turn_id"]

        import asyncio
        await asyncio.sleep(0.05)  # let the fire-and-forget background task run

    background_records = [r for r in caplog.records if r.name == "brain.background"]
    assert background_records  # sanity: the background task actually logged something
    assert all(getattr(r, "turn_id", None) == turn_id for r in background_records)


async def test_request_id_header_is_exposed_for_cross_origin_requests(client):
    # Without Access-Control-Expose-Headers, browsers hide X-Request-Id from
    # JS on cross-origin responses even though it's on the wire — regression
    # for local-conversation-sse.ts's describeHttpFailure() reading it via fetch.
    resp = await client.get("/health/live", headers={"Origin": "http://localhost:5173"})
    exposed = resp.headers["access-control-expose-headers"]
    assert "X-Request-Id" in exposed


async def test_unhandled_exception_returns_stable_error_shape_with_request_id(client, fake_brain_bundle, monkeypatch):
    async def _boom(messages):
        raise ValueError("simulated unexpected failure")

    monkeypatch.setattr(fake_brain_bundle.llm, "chat", _boom)

    resp = await client.post("/v1/chat", json={"query": "hello"}, headers={"X-Request-Id": "req-boom"})

    assert resp.status_code == 500
    body = resp.json()
    assert body["error"]["code"] == "INTERNAL_ERROR"
    assert body["error"]["request_id"] == "req-boom"
    assert "ValueError" not in body["error"]["message"]  # internals not leaked to the client
