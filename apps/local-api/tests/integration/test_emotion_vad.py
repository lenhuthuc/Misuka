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


async def test_emotion_vad_abandons_background_llm_work_before_the_turn_arrives(
    fake_brain_bundle, client, monkeypatch,
):
    """ROOT CAUSE: background work was only interrupted once `/v1/chat` opened
    the gate — a whole transcription later. By then it had been holding Ollama's
    single runner for the entire duration of Whisper.

    This request *is* the user speaking, so it is the earliest honest signal.
    """
    import asyncio

    gate = fake_brain_bundle.llm_gate
    progress: list[str] = []

    async def slow_background_work():
        progress.append("started")
        await asyncio.sleep(5.0)
        progress.append("should-not-reach")

    runner = asyncio.create_task(gate.run_when_idle(slow_background_work, timeout=2.0))
    await asyncio.sleep(0.05)
    assert progress == ["started"]

    resp = await client.post(
        "/emotion-vad",
        files={"audio": ("segment.wav", make_wav_bytes(), "audio/wav")},
    )

    assert resp.status_code == 200
    assert await runner is False
    assert progress == ["started"]


async def test_emotion_vad_drops_a_playback_reservation_it_interrupted(fake_brain_bundle, client):
    """@example: the user talks over a long reply -> the reservation for audio
    that barge-in already stopped does not keep blocking curation."""
    gate = fake_brain_bundle.llm_gate
    gate.hold_active(300.0)

    resp = await client.post(
        "/emotion-vad",
        files={"audio": ("segment.wav", make_wav_bytes(), "audio/wav")},
    )

    assert resp.status_code == 200
    assert await gate.wait_until_idle(timeout=1.0) is True
