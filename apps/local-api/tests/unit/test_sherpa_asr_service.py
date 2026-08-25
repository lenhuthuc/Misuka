import numpy as np
import pytest

from service.sherpa_asr_service import (
    SherpaASRService,
    SherpaModelFilesMissing,
    SherpaRecognizerInitError,
    normalize_transcript,
)


class _FakeResult:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeStream:
    def __init__(self, recognizer: "_FakeRecognizer") -> None:
        self._recognizer = recognizer
        self.accepted: tuple[int, np.ndarray] | None = None
        self.result = _FakeResult("")

    def accept_waveform(self, sample_rate: int, samples: np.ndarray) -> None:
        self.accepted = (sample_rate, samples)


class _FakeRecognizer:
    def __init__(self) -> None:
        self.streams: list[_FakeStream] = []
        self.decode_calls = 0

    def create_stream(self) -> _FakeStream:
        stream = _FakeStream(self)
        self.streams.append(stream)
        return stream

    def decode_stream(self, stream: _FakeStream) -> None:
        self.decode_calls += 1
        stream.result = _FakeResult("  xin   chào  ")


class _FakeOfflineRecognizer:
    from_transducer_calls = 0

    @classmethod
    def from_transducer(cls, **kwargs):
        cls.from_transducer_calls += 1
        cls.last_kwargs = kwargs
        return _FakeRecognizer()


@pytest.fixture
def model_files(tmp_path):
    paths = {}
    for name in ("tokens.txt", "encoder.onnx", "decoder.onnx", "joiner.onnx"):
        p = tmp_path / name
        p.write_text("x")
        paths[name] = str(p)
    return paths


@pytest.fixture(autouse=True)
def fake_sherpa_onnx(monkeypatch):
    import service.sherpa_asr_service as mod

    _FakeOfflineRecognizer.from_transducer_calls = 0
    monkeypatch.setattr(mod.sherpa_onnx, "OfflineRecognizer", _FakeOfflineRecognizer)
    return _FakeOfflineRecognizer


def test_missing_model_files_raise_a_clear_error(tmp_path):
    with pytest.raises(SherpaModelFilesMissing):
        SherpaASRService(
            tokens=str(tmp_path / "missing-tokens.txt"),
            encoder=str(tmp_path / "missing-encoder.onnx"),
            decoder=str(tmp_path / "missing-decoder.onnx"),
            joiner=str(tmp_path / "missing-joiner.onnx"),
        )


def test_recognizer_is_constructed_exactly_once(model_files, fake_sherpa_onnx):
    SherpaASRService(**{k.split(".")[0]: v for k, v in model_files.items()})
    assert fake_sherpa_onnx.from_transducer_calls == 1


def test_recognizer_construction_failure_is_wrapped(model_files, monkeypatch):
    import service.sherpa_asr_service as mod

    class _Boom:
        @classmethod
        def from_transducer(cls, **kwargs):
            raise RuntimeError("bad onnx file")

    monkeypatch.setattr(mod.sherpa_onnx, "OfflineRecognizer", _Boom)
    with pytest.raises(mod.SherpaRecognizerInitError):
        SherpaASRService(**{k.split(".")[0]: v for k, v in model_files.items()})


def test_transcribe_normalizes_whitespace(model_files):
    kwargs = {k.split(".")[0]: v for k, v in model_files.items()}
    service = SherpaASRService(**kwargs)
    samples = np.zeros(16000, dtype=np.float32)

    text = service.transcribe(samples)

    assert text == "Xin chào"


def test_normalize_transcript_folds_the_models_all_caps_output_to_sentence_case():
    """ROOT CAUSE: this recognizer's whole BPE vocabulary is uppercase, so every
    transcript shouted. Measured on qwen2.5:1.5b and :3b, "ĐỐ BẠN LÀ MỘT CỘNG
    MỘT BẰNG BAO NHIÊU" was answered as a question about the assistant's
    identity by both models; the same sentence lowercased was answered
    "một cộng một bằng hai" by both.
    """
    assert normalize_transcript("ĐỐ BẠN LÀ MỘT CỘNG MỘT BẰNG BAO NHIÊU") == (
        "Đố bạn là một cộng một bằng bao nhiêu"
    )


def test_normalize_transcript_capitalises_past_leading_punctuation():
    """@example: the recognizer can emit a leading quote or dash, and
    `str.capitalize()` would leave the first *word* lowercase in that case."""
    assert normalize_transcript('"XIN CHÀO"') == '"Xin chào"'


def test_normalize_transcript_leaves_a_digitsonly_transcript_alone():
    assert normalize_transcript("123") == "123"


def test_transcribe_feeds_the_full_waveform_without_trimming(model_files):
    kwargs = {k.split(".")[0]: v for k, v in model_files.items()}
    service = SherpaASRService(**kwargs)
    samples = np.zeros(160000, dtype=np.float32)  # 10s @ 16kHz

    service.transcribe(samples)

    stream = service._recognizer.streams[-1]
    assert stream.accepted[1].shape[0] == 160000
