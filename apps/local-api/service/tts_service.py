"""Text-to-speech over two engines behind one voice registry.

Piper is fast but flat: its voices come from read-speech corpora and the
architecture carries no emotion conditioning, so every voice sounds like a
newsreader. Kokoro is roughly six times heavier — measured RTF 0.34 against
Piper's 0.057 on this CPU, both far under real time — and buys genuine vocal
character. Neither replaces the other: Kokoro has no Vietnamese, so Piper stays
for that, and the choice is per voice id rather than global.

Synthesis is deliberately synchronous here and pushed to a thread by callers.
At Kokoro's ~2s per reply, running it on the event loop would stall every other
request, including an in-flight chat stream.
"""

import io
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from piper import PiperVoice


def _wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    """Pack mono float32 samples in [-1, 1] into a 16-bit PCM WAV container."""
    clipped = np.clip(samples, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype("<i2")

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())
    return buf.getvalue()


@dataclass
class _PiperVoiceEntry:
    id: str
    path: Path
    config: Path
    _voice: PiperVoice | None = None

    def render_wav(self, text: str) -> bytes:
        if self._voice is None:
            print(f"[Piper] Loading voice '{self.id}' from {self.path.name} ...")
            self._voice = PiperVoice.load(str(self.path), config_path=str(self.config))
            print(f"[Piper] Voice '{self.id}' ready (sample_rate={self._voice.config.sample_rate})")

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            self._voice.synthesize_wav(text, wf)
        return buf.getvalue()


def _discover_piper_voices(models_dir: Path) -> dict[str, _PiperVoiceEntry]:
    """Pair every `.onnx` under `models_dir` with its `.onnx.json` sidecar.

    Before:
    - "vi_VN-25hours_single-low.onnx"

    After:
    - voice id "25hours_single-low"
    """
    voices: dict[str, _PiperVoiceEntry] = {}
    for onnx_file in sorted(models_dir.glob("*.onnx")):
        config_file = onnx_file.with_suffix(".onnx.json")
        if not config_file.exists():
            continue
        stem = onnx_file.stem
        parts = stem.split("-", 1)
        voice_id = parts[1] if len(parts) > 1 else stem
        voices[voice_id] = _PiperVoiceEntry(id=voice_id, path=onnx_file, config=config_file)
    return voices


class _KokoroBackend:
    """Lazily-loaded Kokoro engine; one 310MB model serves all of its voices.

    Loading is deferred because the model costs about a second and a deployment
    may never ask for a Kokoro voice.
    """

    def __init__(self, model_path: Path, voices_path: Path) -> None:
        self._model_path = model_path
        self._voices_path = voices_path
        self._kokoro = None

    def available(self) -> bool:
        return self._model_path.exists() and self._voices_path.exists()

    def voice_ids(self) -> list[str]:
        """Voice names read without loading the model: the embeddings file is a
        plain npz, so listing costs nothing at startup."""
        if not self.available():
            return []
        with np.load(self._voices_path) as data:
            return sorted(data.files)

    def render_wav(self, voice_id: str, text: str) -> bytes:
        if self._kokoro is None:
            from kokoro_onnx import Kokoro

            print(f"[Kokoro] Loading {self._model_path.name} ...")
            self._kokoro = Kokoro(str(self._model_path), str(self._voices_path))
            print("[Kokoro] Ready.")

        samples, sample_rate = self._kokoro.create(text, voice=voice_id, speed=1.0, lang="en-us")
        return _wav_bytes(np.asarray(samples), sample_rate)


class TTSService:
    """Resolves a voice id to whichever engine owns it and renders WAV bytes.

    Use when: the app needs text-to-speech. Construct exactly once, inside the
    service container.

    Expects: `synthesize_wav` to be called off the event loop — it blocks for
    as long as synthesis takes.

    Returns: complete 16-bit PCM WAV bytes, ready to stream to a client.
    """

    def __init__(self, piper_models_dir: Path, kokoro_models_dir: Path | None = None) -> None:
        self._piper = _discover_piper_voices(piper_models_dir)

        self._kokoro: _KokoroBackend | None = None
        kokoro_ids: list[str] = []
        if kokoro_models_dir is not None:
            backend = _KokoroBackend(
                kokoro_models_dir / "kokoro-v1.0.onnx",
                kokoro_models_dir / "voices-v1.0.bin",
            )
            if backend.available():
                self._kokoro = backend
                kokoro_ids = backend.voice_ids()

        # Piper first so a name present in both keeps its existing meaning.
        self._kokoro_ids = [vid for vid in kokoro_ids if vid not in self._piper]

        if not self._piper and not self._kokoro_ids:
            print(f"[TTS] WARNING: no voice found in {piper_models_dir} or {kokoro_models_dir}")
        else:
            print(f"[TTS] Piper: {list(self._piper)}")
            print(f"[TTS] Kokoro: {len(self._kokoro_ids)} voice(s)")

    def list_voices(self) -> list[dict]:
        return (
            [{"id": vid, "name": vid, "engine": "piper"} for vid in self._piper]
            + [{"id": vid, "name": vid, "engine": "kokoro"} for vid in self._kokoro_ids]
        )

    def has_voice(self, voice_id: str) -> bool:
        return voice_id in self._piper or voice_id in self._kokoro_ids

    def synthesize_wav(self, voice_id: str, text: str) -> bytes:
        if entry := self._piper.get(voice_id):
            return entry.render_wav(text)
        if self._kokoro is not None and voice_id in self._kokoro_ids:
            return self._kokoro.render_wav(voice_id, text)
        raise KeyError(voice_id)
