from service.whisper_service import WhisperService


class _Segment:
    text = " Xin chào "
    no_speech_prob = 0.0


class _FakeModel:
    def __init__(self) -> None:
        self.kwargs = {}

    def transcribe(self, _audio_path: str, **kwargs):
        self.kwargs = kwargs
        return iter([_Segment()]), object()


def test_english_transcription_uses_fast_decode_without_biasing_prompt():
    service = object.__new__(WhisperService)
    service._model = _FakeModel()

    assert service.transcribe("voice.wav", language="en") == "Xin chào"
    assert service._model.kwargs["beam_size"] == 1
    assert service._model.kwargs["initial_prompt"] is None
    assert service._model.kwargs["without_timestamps"] is True
    assert service._model.kwargs["vad_parameters"]["min_silence_duration_ms"] == 500


def test_explicit_prompt_overrides_default_language_prompt():
    service = object.__new__(WhisperService)
    service._model = _FakeModel()

    service.transcribe("voice.wav", language="en", prompt="Mitsuka")

    assert service._model.kwargs["initial_prompt"] == "Mitsuka"


def test_legitimate_short_speaking_practice_phrase_is_not_filtered():
    service = object.__new__(WhisperService)
    service._model = _FakeModel()
    service._model.transcribe = lambda *_args, **_kwargs: (
        iter([type("Segment", (), {"text": " Thank you. ", "no_speech_prob": 0.1})()]),
        object(),
    )

    assert service.transcribe("voice.wav", language="en") == "Thank you."
