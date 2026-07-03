"""
POST /emotion-vad

Nhận file audio, chạy song song:
  Branch 1: wav2vec2 → audio VAD (v, a, d) ∈ [-1, 1]
  Branch 2: faster-whisper → transcript → PhoBERT → text VAD (v, a, d) ∈ [-1, 1]

Fused = audio × 0.7 + text × 0.3
"""

import asyncio
import io
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor

import librosa
import numpy as np
import soundfile as sf
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from schemas.vad import EmotionVADResponse, VADScores
from service.audio_emotion_service import AudioEmotionService
from service.vad_service import VADService
from service.whisper_service import transcribe

router = APIRouter(prefix="/emotion-vad", tags=["Emotion VAD"])

_executor = ThreadPoolExecutor(max_workers=4)

_AUDIO_WEIGHT = 0.7
_TEXT_WEIGHT  = 0.3


# ── Dependency injectors (overridden in main.py) ──────────────────────────────

def get_vad_service() -> VADService:
    raise NotImplementedError

def get_audio_service() -> AudioEmotionService:
    raise NotImplementedError


# ── Helpers ───────────────────────────────────────────────────────────────────

def _scale_to_signed(v: float, a: float, d: float) -> VADScores:
    """PhoBERT outputs [0, 1]. Convert to [-1, 1]."""
    return VADScores(valence=v , arousal=a, dominance=d)


def _fuse(audio: VADScores, text: VADScores) -> VADScores:
    return VADScores(
        valence=audio.valence * _AUDIO_WEIGHT + text.valence * _TEXT_WEIGHT,
        arousal=audio.arousal * _AUDIO_WEIGHT + text.arousal * _TEXT_WEIGHT,
        dominance=audio.dominance * _AUDIO_WEIGHT + text.dominance * _TEXT_WEIGHT,
    )


# ── Route ─────────────────────────────────────────────────────────────────────

@router.post("", response_model=EmotionVADResponse)
async def emotion_vad(
    audio_file: UploadFile = File(..., alias="audio"),
    vad_svc: VADService = Depends(get_vad_service),
    audio_svc: AudioEmotionService = Depends(get_audio_service),
) -> EmotionVADResponse:
    raw = await audio_file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty audio file.")

    try:
        audio_array, orig_sr = sf.read(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Cannot decode audio: {exc}") from exc

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

        async def _text_branch() -> tuple[str, VADScores]:
            # ASR → transcript → PhoBERT VAD (sequential within this branch)
            transcript: str = await loop.run_in_executor(
                _executor, transcribe, tmp.name
            )
            v, a, d = await loop.run_in_executor(
                _executor, vad_svc.predict, transcript
            )
            return transcript, _scale_to_signed(v, a, d)

        audio_task = loop.run_in_executor(
            _executor, audio_svc.predict, audio_array
        )
        text_task = _text_branch()

        # Both branches run concurrently
        (av, aa, ad), (transcript, text_vad) = await asyncio.gather(
            audio_task, text_task
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
