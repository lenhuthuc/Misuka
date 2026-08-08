import os
from pathlib import Path

from faster_whisper import WhisperModel
from huggingface_hub import snapshot_download

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
        # CTranslate2 otherwise chooses a conservative CPU thread count. Voice
        # input is latency-sensitive, so use the available logical cores while
        # capping the value to avoid pathological oversubscription.
        cpu_threads = min(os.cpu_count() or 4, 16) if device == "cpu" else 0
        self._model = WhisperModel(
            str(model_local),
            device=device,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
        )

    def transcribe(
        self,
        audio_path: str,
        language: str | None = "en",
        prompt: str | None = None,
    ) -> str:
        """Transcribe audio file, return plain text."""
        segments, _info = self._model.transcribe(
            audio_path,
            language=language,
            # Greedy decoding is substantially faster than a five-candidate
            # beam on CPU and is accurate enough for interactive voice input.
            beam_size=1,
            temperature=0,                     # greedy decoding, no randomness
            condition_on_previous_text=False,  # prevent hallucination loops
            no_speech_threshold=0.6,           # was 0.9 — standard recommended threshold
            log_prob_threshold=-1.0,           # drop low-confidence segments
            vad_filter=True,
            # Interactive recordings often contain silence before/after the
            # utterance. Trim it aggressively so Whisper neither spends time
            # on it nor hallucinates text from background noise.
            vad_parameters={
                "min_silence_duration_ms": 500,
                "speech_pad_ms": 150,
            },
            without_timestamps=True,
            initial_prompt=prompt,
        )

        parts = []
        for seg in segments:
            text = seg.text.strip()
            if not text:
                continue
            # Drop segments Whisper is not confident about
            if seg.no_speech_prob > 0.6:
                continue
            parts.append(text)

        return " ".join(parts)
