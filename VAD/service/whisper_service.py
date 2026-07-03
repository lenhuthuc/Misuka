import os
from pathlib import Path

from faster_whisper import WhisperModel
from huggingface_hub import snapshot_download

_MODEL_SIZE = os.getenv("WHISPER_MODEL", "medium.en")
_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
_COMPUTE = os.getenv("WHISPER_COMPUTE", "int8")

_MODELS_DIR = Path(__file__).parent.parent / "models"
_MODEL_LOCAL = _MODELS_DIR / _MODEL_SIZE

print(f"[Whisper] Loading model '{_MODEL_SIZE}' on {_DEVICE} ({_COMPUTE}) ...")
if not _MODEL_LOCAL.exists():
    print(f"[Whisper] Downloading to {_MODEL_LOCAL} ...")
    snapshot_download(
        repo_id=f"Systran/faster-whisper-{_MODEL_SIZE}",
        local_dir=str(_MODEL_LOCAL),
    )
whisper_model = WhisperModel(str(_MODEL_LOCAL), device=_DEVICE, compute_type=_COMPUTE)
print("[Whisper] Model ready.")


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


def transcribe(audio_path: str, language: str | None = "en") -> str:
    """Transcribe audio file, return plain text."""
    segments, _info = whisper_model.transcribe(
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
