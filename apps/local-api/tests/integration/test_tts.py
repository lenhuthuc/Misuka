async def test_tts_speech_success(client):
    resp = await client.post("/v1/audio/speech", json={"input": "xin chao"})
    assert resp.status_code == 200
    assert resp.content[:4] == b"RIFF"  # WAV container magic bytes


async def test_tts_speech_empty_input_returns_422_with_stable_code(client):
    resp = await client.post("/v1/audio/speech", json={"input": "   "})
    assert resp.status_code == 422
    assert resp.json()["detail"]["code"] == "TTS_EMPTY_INPUT"


async def test_tts_speech_unknown_voice_returns_404_with_stable_code(client):
    resp = await client.post("/v1/audio/speech", json={"input": "xin chao", "voice": "no-such-voice"})
    assert resp.status_code == 404
    assert resp.json()["detail"]["code"] == "TTS_UNKNOWN_VOICE"


async def test_tts_speech_known_voice_succeeds(client):
    resp = await client.post("/v1/audio/speech", json={"input": "xin chao", "voice": "fake-voice"})
    assert resp.status_code == 200


async def test_tts_list_voices(client):
    resp = await client.get("/v1/audio/voices")
    assert resp.status_code == 200
    assert resp.json() == {"voices": ["fake-voice"]}


async def test_tts_stream_returns_full_wav_payload(client):
    resp = await client.post("/v1/audio/speech/stream", json={"input": "xin chao"})
    assert resp.status_code == 200
    assert resp.content[:4] == b"RIFF"
    assert len(resp.content) > 4096  # spans multiple 4096-byte send() chunks server-side


async def test_tts_stream_unknown_voice_returns_404_with_stable_code(client):
    resp = await client.post("/v1/audio/speech/stream", json={"input": "xin chao", "voice": "no-such-voice"})
    assert resp.status_code == 404
    assert resp.json()["detail"]["code"] == "TTS_UNKNOWN_VOICE"
