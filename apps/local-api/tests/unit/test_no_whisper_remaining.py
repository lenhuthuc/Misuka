"""Regression: Whisper must be fully replaced by Sherpa-ONNX — no production
module should import or construct it any more."""
from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_PRODUCTION_DIRS = ("api", "service", "model", "brain", "core", "schemas", "application")
_NEEDLES = ("faster_whisper", "WhisperModel", "openai-whisper", "openai_whisper")


def _production_py_files():
    for dir_name in _PRODUCTION_DIRS:
        yield from (_ROOT / dir_name).rglob("*.py")
    yield _ROOT / "main.py"


def test_no_production_module_imports_or_constructs_whisper():
    offenders = []
    for path in _production_py_files():
        text = path.read_text(encoding="utf-8")
        for needle in _NEEDLES:
            if needle in text:
                offenders.append((str(path.relative_to(_ROOT)), needle))
    assert offenders == []


def test_requirements_no_longer_pin_a_whisper_package():
    text = (_ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "faster-whisper" not in text
    assert "openai-whisper" not in text


def test_service_container_has_no_whisper_attribute():
    from core.container import ServiceContainer

    field_names = {f for f in ServiceContainer.__dataclass_fields__}
    assert "whisper" not in field_names
    assert "audio_emotion" not in field_names
    assert {"asr", "text_vad", "multimodal_vad", "emotion_pipeline"} <= field_names
