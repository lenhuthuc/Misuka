import pytest

import api.emotion_vad as emotion_vad_api
from tests.conftest import make_wav_bytes


@pytest.fixture(autouse=True)
def debug_audio_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(emotion_vad_api, "_DEBUG_AUDIO_DIR", tmp_path)
    return tmp_path


async def test_emotion_vad_success(client):
    wav = make_wav_bytes()
    resp = await client.post(
        "/emotion-vad",
        files={"audio": ("segment.wav", wav, "audio/wav")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["transcript"] == "fake transcript"
    assert body["asr"] == {"engine": "sherpa-onnx", "language": "vi"}
    assert body["user_vad"]["mode"] == "multimodal"
    for key in ("valence", "arousal", "dominance"):
        assert 0.0 <= body["user_vad"][key] <= 1.0


async def test_emotion_vad_multimodal_branch_gets_a_fixed_length_center_crop(client, fake_brain_bundle):
    """Regression: WavLM must always see the checkpoint's trained window
    length (4s @ 16kHz = 64000 samples) — shorter clips are zero-padded, never
    fed in at their native (shorter) length."""
    wav = make_wav_bytes(duration_sec=0.5)  # far shorter than 4s

    resp = await client.post(
        "/emotion-vad",
        files={"audio": ("segment.wav", wav, "audio/wav")},
    )

    assert resp.status_code == 200
    audio_len, mask_len, text = fake_brain_bundle.multimodal_vad.calls[-1]
    assert audio_len == 64000
    assert mask_len == 64000
    assert text == "fake transcript"


async def test_emotion_vad_saves_exact_upload_before_decoding(client, debug_audio_dir):
    wav = make_wav_bytes()

    resp = await client.post(
        "/emotion-vad",
        files={"audio": ("segment.wav", wav, "audio/wav")},
    )

    assert resp.status_code == 200
    assert (debug_audio_dir / "last-audio-input.wav").read_bytes() == wav


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


async def test_emotion_vad_drops_a_playback_reservation_it_interrupted(fake_brain_bundle, client):
    """@example: the user talks over a long reply -> the reservation for audio
    that barge-in already stopped is dropped, rather than holding exchange
    indexing back for the five minutes it had been promised."""
    gate = fake_brain_bundle.llm_gate
    gate.hold_active(300.0)
    assert gate._playback_held is True

    resp = await client.post(
        "/emotion-vad",
        files={"audio": ("segment.wav", make_wav_bytes(), "audio/wav")},
    )

    assert resp.status_code == 200
    assert gate._playback_held is False
    assert gate._playback_until == 0.0
