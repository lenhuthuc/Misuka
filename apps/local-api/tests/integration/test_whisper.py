from tests.conftest import make_wav_bytes


async def test_transcriptions_defaults_to_english_when_omitted(client, fake_brain_bundle):
    wav = make_wav_bytes()
    resp = await client.post(
        "/v1/audio/transcriptions",
        files={"file": ("segment.wav", wav, "audio/wav")},
    )

    assert resp.status_code == 200
    assert fake_brain_bundle.whisper.calls[-1][1] == "en"


async def test_transcriptions_returns_text(client):
    wav = make_wav_bytes()
    resp = await client.post(
        "/v1/audio/transcriptions",
        files={"file": ("segment.wav", wav, "audio/wav")},
        data={"model": "whisper-1", "language": "vi"},
    )
    assert resp.status_code == 200
    assert resp.json() == {"text": "fake transcript"}


async def test_transcriptions_text_format_returns_plain_string(client):
    wav = make_wav_bytes()
    resp = await client.post(
        "/v1/audio/transcriptions",
        files={"file": ("segment.wav", wav, "audio/wav")},
        data={"response_format": "text"},
    )
    assert resp.status_code == 200
    assert resp.text == '"fake transcript"'
