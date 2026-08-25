from tests.conftest import make_wav_bytes


async def test_transcriptions_returns_text(client):
    wav = make_wav_bytes()
    resp = await client.post(
        "/v1/audio/transcriptions",
        files={"file": ("segment.wav", wav, "audio/wav")},
        data={"model": "whisper-1"},
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


async def test_transcriptions_transcribes_the_full_uploaded_audio(client, fake_brain_bundle):
    """Regression for the Whisper -> Sherpa-ONNX swap: the ASR branch must
    never be trimmed to any fixed duration before decoding."""
    wav = make_wav_bytes(duration_sec=2.0)
    resp = await client.post(
        "/v1/audio/transcriptions",
        files={"file": ("segment.wav", wav, "audio/wav")},
    )
    assert resp.status_code == 200
    assert fake_brain_bundle.asr.calls[-1] == 32000  # 2.0s @ 16kHz, not cropped


async def test_transcriptions_undecodable_audio_returns_422(client):
    resp = await client.post(
        "/v1/audio/transcriptions",
        files={"file": ("segment.wav", b"not a real wav file", "audio/wav")},
    )
    assert resp.status_code == 422
