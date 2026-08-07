"""App startup/shutdown and health endpoints.

/health/ready only asserts required in-process deps constructed — see
REFACTOR_PLAN.md Phase 3. /health is kept as an alias of /health/live for the
existing Stage.vue local-server probe.
"""


async def test_health_live(client):
    resp = await client.get("/health/live")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


async def test_health_alias_matches_live(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


async def test_health_ready(client):
    resp = await client.get("/health/ready")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


async def test_app_startup_wires_container(client, fake_brain_bundle):
    # The lifespan ran (client fixture entered it) — app.state.container should be populated.
    import main

    container = main.app.state.container
    assert container.llm is fake_brain_bundle.llm
    assert container.memory is fake_brain_bundle.memory
    assert container.vector is fake_brain_bundle.vector
    assert container.rag is fake_brain_bundle.rag
    assert container.vad is fake_brain_bundle.vad
    assert container.audio_emotion is fake_brain_bundle.audio_emotion
    assert container.whisper is fake_brain_bundle.whisper
    assert container.tts is fake_brain_bundle.tts


async def test_models_list_endpoint(client):
    resp = await client.get("/v1/models")
    assert resp.status_code == 200
    assert resp.json()["data"][0]["id"] == "whisper-1"
