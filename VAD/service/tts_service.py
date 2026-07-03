"""Piper TTS service — load ONNX voice models từ thư mục cấu hình."""

import os
from dataclasses import dataclass
from pathlib import Path

from piper import PiperVoice

# Thư mục chứa file *.onnx (mặc định: thư mục cha của VAD/)
_MODELS_DIR = Path(
    os.getenv("PIPER_MODELS_DIR", str(Path(__file__).resolve().parents[2]))
)


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


# Registry khởi tạo khi module load — model được load lazy khi dùng lần đầu
_VOICES: dict[str, VoiceEntry] = _discover_voices(_MODELS_DIR)

if not _VOICES:
    print(f"[Piper] WARNING: No .onnx model found in {_MODELS_DIR}")
else:
    print(f"[Piper] Found {len(_VOICES)} voice(s): {list(_VOICES.keys())}")


def list_voices() -> list[dict]:
    return [{"id": v.id, "name": v.id} for v in _VOICES.values()]


def get_voice(voice_id: str) -> PiperVoice:
    """Lấy PiperVoice theo id, load nếu chưa load."""
    if voice_id not in _VOICES:
        # Fallback về voice đầu tiên nếu không tìm thấy
        if not _VOICES:
            raise ValueError("Không có Piper model nào được cài đặt.")
        voice_id = next(iter(_VOICES))
    return _VOICES[voice_id].load()


def get_sample_rate(voice_id: str) -> int:
    if voice_id not in _VOICES:
        voice_id = next(iter(_VOICES))
    return _VOICES[voice_id].sample_rate
