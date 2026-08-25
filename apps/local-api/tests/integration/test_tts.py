import pytest


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


async def test_speech_reserves_the_reply_playback_on_the_llm_gate(fake_brain_bundle, client):
    """ROOT CAUSE: exchange indexing started the moment the last token was
    generated, which is when Piper starts rendering the reply, so the embedding
    and the synthesis fought over the same cores.

    The synthesised clip is what tells the gate how long that reply occupies
    the user for.
    """
    gate = fake_brain_bundle.llm_gate
    before = gate._playback_until

    resp = await client.post("/v1/audio/speech", json={"input": "xin chao"})

    assert resp.status_code == 200
    # FakeTTSService renders 20000 frames at 16 kHz.
    assert gate._playback_until - before >= 1.25


async def test_speech_without_vad_renders_with_the_voice_defaults(fake_brain_bundle, client):
    """A caller that sends no emotion is not guessing a neutral one for it —
    it gets exactly the audio Piper would have produced before prosody existed.
    """
    resp = await client.post("/v1/audio/speech", json={"input": "xin chao"})

    assert resp.status_code == 200
    assert fake_brain_bundle.tts.last_plan is None


async def test_speech_with_full_vad_builds_a_prosody_plan(fake_brain_bundle, client):
    resp = await client.post("/v1/audio/speech", json={
        "input": "xin chao", "valence": 0.8, "arousal": 0.9, "dominance": 0.7,
    })

    assert resp.status_code == 200
    plan = fake_brain_bundle.tts.last_plan
    assert plan is not None
    # High arousal: faster than neutral, at the voice's own fixed pitch.
    assert plan.length_scale < 1.0
    assert plan.pitch_scale == pytest.approx(fake_brain_bundle.tts_pitch_scale)
    assert plan.contour_depth == 0.0


async def test_speech_ignores_a_partial_vad_reading(fake_brain_bundle, client):
    """Two axes out of three is not a reading. Treating the missing one as
    neutral would silently speak a half-derived emotion."""
    resp = await client.post("/v1/audio/speech", json={
        "input": "xin chao", "valence": 0.8, "arousal": 0.9,
    })

    assert resp.status_code == 200
    assert fake_brain_bundle.tts.last_plan is None


async def test_speech_rejects_out_of_range_vad(client):
    """The prosody model reads the checkpoints' native [0, 1]; a signed [-1, 1]
    value arriving here would be silently interpreted as "as low as possible"."""
    resp = await client.post("/v1/audio/speech", json={
        "input": "xin chao", "valence": -0.5, "arousal": 0.9, "dominance": 0.7,
    })

    assert resp.status_code == 422


async def test_explicit_speed_still_overrides_the_emotion_derived_tempo(fake_brain_bundle, client):
    """`speed` is an OpenAI-compatible parameter a client sets deliberately.
    Letting the emotion model quietly win would make it a lie."""
    resp = await client.post("/v1/audio/speech", json={
        "input": "xin chao", "speed": 2.0, "valence": 0.5, "arousal": 0.5, "dominance": 0.5,
    })

    assert resp.status_code == 200
    # Neutral V/A/D gives length_scale 1.0; speed 2.0 halves it.
    assert fake_brain_bundle.tts.last_plan.length_scale == 0.5


async def test_speech_stream_route_is_gone(client):
    """Sentence-level streaming is what re-spoke every reply from the top; the
    route existed only to serve it."""
    resp = await client.post("/v1/audio/speech/stream", json={"input": "xin chao"})
    assert resp.status_code == 404


async def test_the_reply_is_spoken_at_one_pitch_whatever_the_sentence_says(fake_brain_bundle, client):
    """ROOT CAUSE: pitch was derived from the utterance's own V/A/D. Once a
    reply was spoken sentence by sentence, each sentence was scored separately
    and came back at its own pitch, so one answer alternated between a low
    voice and a high one. Pitch is now a constant of the voice.
    """
    pitches = []
    for text in ["Mình vui lắm đấy!", "Chuyện đó buồn thật."]:
        resp = await client.post("/v1/audio/speech", json={"input": text, "auto_prosody": True})
        assert resp.status_code == 200
        pitches.append(fake_brain_bundle.tts.last_plan.pitch_scale)

    assert pitches[0] == pytest.approx(pitches[1])
    assert pitches[0] == pytest.approx(fake_brain_bundle.tts_pitch_scale)


async def test_speech_derives_prosody_from_the_text_when_asked(fake_brain_bundle, client):
    """ROOT CAUSE: the reply is spoken one sentence at a time now, and a client
    in the middle of a stream has no V/A/D to send -- the reply's own reading
    is only inferred once the whole reply exists. Without this the entire
    conversation would be spoken in the voice's flat default delivery, which is
    what the first sentence-by-sentence attempt sounded like.
    """
    resp = await client.post("/v1/audio/speech", json={"input": "xin chao", "auto_prosody": True})

    assert resp.status_code == 200
    assert fake_brain_bundle.tts.last_plan is not None


async def test_an_explicit_reading_outranks_the_derived_one(fake_brain_bundle, client):
    """@example: a caller that measured the emotion sends it -> the text model is
    not consulted, and the plan is the one the caller's reading implies."""
    resp = await client.post("/v1/audio/speech", json={
        "input": "xin chao", "valence": 0.8, "arousal": 0.9, "dominance": 0.7, "auto_prosody": True,
    })

    assert resp.status_code == 200
    plan = fake_brain_bundle.tts.last_plan
    assert plan is not None
    # The caller's high arousal speeds the delivery up; the text model's
    # neutral reading would have left the tempo at 1.0.
    assert plan.length_scale < 1.0


async def test_speech_finished_releases_work_waiting_on_playback(fake_brain_bundle, client):
    """@example: the client's playback queue drained -> the exchange indexing that
    was staying out of synthesis's way is released, without waiting out the
    reservation the synthesised clips left behind."""
    gate = fake_brain_bundle.llm_gate
    gate.mark_active()
    assert await gate.wait_until_spoken(timeout=0.05) is False

    resp = await client.post("/v1/audio/speech/finished")

    assert resp.status_code == 204
    assert await gate.wait_until_spoken(timeout=1.0) is True
