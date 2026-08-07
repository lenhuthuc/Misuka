from pathlib import Path

from faster_whisper import WhisperModel
from huggingface_hub import snapshot_download

# Whisper hallucinates these phrases on near-silence / noise input.
# Checked against faster-whisper known hallucination patterns.
_HALLUCINATION_PHRASES = {
    "thank you",
    "thank you.",
    "thank you very much",
    "thank you very much.",
    "thanks for watching",
    "thanks for watching.",
    "thanks.",
    "bye",
    "bye.",
    "bye bye",
    "bye bye.",
    "please subscribe",
    "you",
}


class WhisperService:
    """Loads a faster-whisper model once at construction and transcribes audio files.

    Use when: the app needs speech-to-text. Construct exactly once (inside
    the FastAPI lifespan/service container) — loading is I/O- and CPU-heavy.

    Expects: `models_dir / model_size` either already contains the model, or
    it will be downloaded there on first construction.
    """

    def __init__(self, model_size: str, device: str, compute_type: str, models_dir: Path) -> None:
        model_local = models_dir / model_size
        if not model_local.exists():
            snapshot_download(repo_id=f"Systran/faster-whisper-{model_size}", local_dir=str(model_local))
        self._model = WhisperModel(str(model_local), device=device, compute_type=compute_type)

    def transcribe(self, audio_path: str, language: str | None = "en") -> str:
        """Transcribe audio file, return plain text."""
        segments, _info = self._model.transcribe(
            audio_path,
            language=language,
            beam_size=5,
            temperature=0,                     # greedy decoding, no randomness
            condition_on_previous_text=False,  # prevent hallucination loops
            no_speech_threshold=0.6,           # was 0.9 — standard recommended threshold
            log_prob_threshold=-1.0,           # drop low-confidence segments
            vad_filter=True,
        )

        parts = []
        for seg in segments:
            text = seg.text.strip()
            if not text:
                continue
            # Drop segments Whisper is not confident about
            if seg.no_speech_prob > 0.6:
                continue
            # Drop well-known silence hallucinations
            if text.lower().rstrip(".!?,") in _HALLUCINATION_PHRASES or text.lower() in _HALLUCINATION_PHRASES:
                continue
            parts.append(text)

        return " ".join(parts)
