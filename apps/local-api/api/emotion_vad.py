"""
POST /emotion-vad

Nhận file audio, chạy song song:
  Branch 1: wav2vec2 → audio VAD (v, a, d) ∈ [-1, 1]
  Branch 2: faster-whisper → transcript → PhoBERT → text VAD (v, a, d) ∈ [-1, 1]

Fused = audio × 0.7 + text × 0.3
"""

import asyncio
import io
import logging
import os
import tempfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from api.dependencies import get_container
from core.container import ServiceContainer
from core.logging import log_duration
from schemas.vad import EmotionVADResponse, VADScores

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/emotion-vad", tags=["Emotion VAD"])

_AUDIO_WEIGHT = 0.7
_TEXT_WEIGHT  = 0.3
_DEBUG_AUDIO_DIR = Path(__file__).resolve().parents[1] / "debug_audio"


def _save_raw_whisper_input(raw: bytes) -> Path:
    """Persist the exact upload before soundfile/resampling/Whisper touches it."""
    _DEBUG_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    target = _DEBUG_AUDIO_DIR / "last-whisper-input.wav"
    target.write_bytes(raw)
    logger.info("Saved raw Whisper input to %s (%d bytes)", target, len(raw))
    return target


def _scale_to_signed(v: float, a: float, d: float) -> VADScores:
    """PhoBERT outputs [0, 1]. Convert to [-1, 1]."""
    return VADScores(valence=v , arousal=a, dominance=d)


def _fuse(audio: VADScores, text: VADScores) -> VADScores:
    return VADScores(
        valence=audio.valence * _AUDIO_WEIGHT + text.valence * _TEXT_WEIGHT,
        arousal=audio.arousal * _AUDIO_WEIGHT + text.arousal * _TEXT_WEIGHT,
        dominance=audio.dominance * _AUDIO_WEIGHT + text.dominance * _TEXT_WEIGHT,
    )


@router.post("", response_model=EmotionVADResponse)
async def emotion_vad(
    audio_file: UploadFile = File(..., alias="audio"),
    language: str | None = Form("en"),
    container: ServiceContainer = Depends(get_container),
) -> EmotionVADResponse:
    # This request *is* the user talking, and it arrives before the chat turn
    # it will produce. Telling the gate now abandons any background generation
    # while Whisper still has work to do, so the runner is free by the time the
    # turn asks for it — waiting for /v1/chat to open the gate is a step late.
    container.llm_gate.mark_active()

    raw = await audio_file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty audio file.")

    _save_raw_whisper_input(raw)

    try:
        audio_array, orig_sr = sf.read(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Cannot decode audio: {exc}") from exc

    with log_duration(logger, "audio.decode_resample", component="audio"):
        # Mono downmix + float32
        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1)
        audio_array = audio_array.astype(np.float32)

        # Resample to 16 kHz for wav2vec2 and faster-whisper
        if orig_sr != 16000:
            audio_array = librosa.resample(audio_array, orig_sr=orig_sr, target_sr=16000)

    # Write to temp file once — faster-whisper needs a file path
    suffix = os.path.splitext(audio_file.filename or "audio.wav")[1] or ".wav"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        sf.write(tmp.name, audio_array, 16000)
        tmp.close()

        loop = asyncio.get_event_loop()
        executor = container.emotion_executor

        async def _text_branch() -> tuple[str, VADScores]:
            # ASR → transcript → PhoBERT VAD (sequential within this branch)
            with log_duration(logger, "whisper.transcribe", component="stt"):
                transcript: str = await loop.run_in_executor(
                    executor, container.whisper.transcribe, tmp.name, language
                )
            with log_duration(logger, "vad.predict_text", component="emotion"):
                v, a, d = await loop.run_in_executor(
                    executor, container.vad.predict, transcript
                )
            return transcript, _scale_to_signed(v, a, d)

        async def _audio_branch() -> tuple[float, float, float]:
            with log_duration(logger, "vad.predict_audio", component="emotion"):
                return await loop.run_in_executor(
                    executor, container.audio_emotion.predict, audio_array
                )

        # Both branches run concurrently
        (av, aa, ad), (transcript, text_vad) = await asyncio.gather(
            _audio_branch(), _text_branch()
        )
    finally:
        os.unlink(tmp.name)

    audio_vad = VADScores(valence=av, arousal=aa, dominance=ad)  # already in [-1, 1]

    return EmotionVADResponse(
        transcript=transcript,
        audio=audio_vad,
        text=text_vad,
        fused=_fuse(audio_vad, text_vad),
    )
