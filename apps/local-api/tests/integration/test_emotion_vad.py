import pytest

import api.emotion_vad as emotion_vad_api
from tests.conftest import make_wav_bytes


@pytest.fixture(autouse=True)
def debug_audio_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(emotion_vad_api, "_DEBUG_AUDIO_DIR", tmp_path)
    return tmp_path


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


async def test_emotion_vad_saves_exact_upload_before_decoding(client, debug_audio_dir):
    wav = make_wav_bytes()

    resp = await client.post(
        "/emotion-vad",
        files={"audio": ("segment.wav", wav, "audio/wav")},
    )

    assert resp.status_code == 200
    assert (debug_audio_dir / "last-whisper-input.wav").read_bytes() == wav


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
