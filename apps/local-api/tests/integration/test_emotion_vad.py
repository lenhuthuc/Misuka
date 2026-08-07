from tests.conftest import make_wav_bytes


async def test_emotion_vad_forwards_language_to_whisper(client, fake_brain_bundle):
    wav = make_wav_bytes()

    resp = await client.post(
        "/emotion-vad",
        files={"audio": ("segment.wav", wav, "audio/wav")},
        data={"language": "vi"},
    )

    assert resp.status_code == 200
    assert [lang for _path, lang in fake_brain_bundle.whisper.calls] == ["vi"]


async def test_emotion_vad_success(client):
    wav = make_wav_bytes()
    resp = await client.post(
        "/emotion-vad",
        files={"audio": ("segment.wav", wav, "audio/wav")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["transcript"] == "fake transcript"
    for key in ("audio", "text", "fused"):
        assert set(body[key].keys()) == {"valence", "arousal", "dominance"}


async def test_emotion_vad_empty_file_returns_400(client):
    resp = await client.post(
        "/emotion-vad",
        files={"audio": ("segment.wav", b"", "audio/wav")},
    )
    assert resp.status_code == 400


async def test_emotion_vad_undecodable_audio_returns_422(client):
    resp = await client.post(
        "/emotion-vad",
        files={"audio": ("segment.wav", b"not a real wav file", "audio/wav")},
    )
    assert resp.status_code == 422
