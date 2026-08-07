"""Piper TTS service — load ONNX voice models từ thư mục cấu hình."""

from dataclasses import dataclass
from pathlib import Path

from piper import PiperVoice


@dataclass
class VoiceEntry:
    id: str          # tên hiển thị, dùng làm voice id
    path: Path       # đường dẫn .onnx
    config: Path     # đường dẫn .onnx.json
    _voice: PiperVoice | None = None

    def load(self) -> PiperVoice:
        if self._voice is None:
            print(f"[Piper] Loading voice '{self.id}' from {self.path.name} ...")
            self._voice = PiperVoice.load(str(self.path), config_path=str(self.config))
            print(f"[Piper] Voice '{self.id}' ready (sample_rate={self._voice.config.sample_rate})")
        return self._voice

    @property
    def sample_rate(self) -> int:
        return self.load().config.sample_rate


def _discover_voices(models_dir: Path) -> dict[str, VoiceEntry]:
    """Quét thư mục, tìm tất cả cặp .onnx + .onnx.json."""
    voices: dict[str, VoiceEntry] = {}
    for onnx_file in sorted(models_dir.glob("*.onnx")):
        config_file = onnx_file.with_suffix(".onnx.json")
        if not config_file.exists():
            continue
        # vi_VN-25hours_single-low.onnx → voice id = "25hours-low"
        stem = onnx_file.stem  # e.g. "vi_VN-25hours_single-low"
        parts = stem.split("-", 1)
        voice_id = parts[1] if len(parts) > 1 else stem
        voices[voice_id] = VoiceEntry(id=voice_id, path=onnx_file, config=config_file)
    return voices


class TTSService:
    """Discovers Piper voices under `models_dir` once at construction; each
    voice's ONNX weights are then loaded lazily on first use.

    Use when: the app needs text-to-speech. Construct exactly once (inside
    the FastAPI lifespan/service container).
    """

    def __init__(self, models_dir: Path) -> None:
        self._voices = _discover_voices(models_dir)
        if not self._voices:
            print(f"[Piper] WARNING: No .onnx model found in {models_dir}")
        else:
            print(f"[Piper] Found {len(self._voices)} voice(s): {list(self._voices.keys())}")

    def list_voices(self) -> list[dict]:
        return [{"id": v.id, "name": v.id} for v in self._voices.values()]

    def get_voice(self, voice_id: str) -> PiperVoice:
        return self._voices[voice_id].load()

    def has_voice(self, voice_id: str) -> bool:
        return voice_id in self._voices

    def get_sample_rate(self, voice_id: str) -> int:
        return self._voices[voice_id].sample_rate
