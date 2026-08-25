"""POST /emotion-vad

Runs the user-audio side of the VAD architecture (see
`service/emotion_pipeline.py`): Sherpa-ONNX transcribes the full recording,
then WavLM (audio) + PhoBERT (that transcript) jointly produce `user_vad`.
"""
import asyncio
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from api.dependencies import get_container
from core.container import ServiceContainer
from core.logging import log_duration
from schemas.vad import ASRInfo, EmotionVADResponse, UserVAD
from service.audio_preprocessing import AudioDecodeError
from service.emotion_pipeline import UserAudioResult
from service.multimodal_vad_service import VADOutputInvalid
from service.sherpa_asr_service import SherpaModelFilesMissing

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/emotion-vad", tags=["Emotion VAD"])

_DEBUG_AUDIO_DIR = Path(__file__).resolve().parents[1] / "debug_audio"


def _save_raw_upload(raw: bytes) -> Path:
    """Persist the exact upload before decode/resample/ASR touches it."""
    _DEBUG_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    target = _DEBUG_AUDIO_DIR / "last-audio-input.wav"
    target.write_bytes(raw)
    logger.info("Saved raw audio input to %s (%d bytes)", target, len(raw))
    return target


@router.post("", response_model=EmotionVADResponse)
async def emotion_vad(
    audio_file: UploadFile | None = File(None, alias="audio"),
    text: str | None = Form(None),
    container: ServiceContainer = Depends(get_container),
) -> EmotionVADResponse:
    # This request *is* the user talking/typing, and it arrives before the
    # chat turn it will produce. Telling the gate now abandons any background
    # generation while this route still has work to do, so the runner is free
    # by the time the turn asks for it — waiting for /v1/chat is a step late.
    container.llm_gate.mark_active()

    raw = await audio_file.read() if audio_file is not None else b""
    if not raw and not text:
        raise HTTPException(status_code=400, detail="Provide either an 'audio' file or 'text'.")

    loop = asyncio.get_event_loop()

    if raw:
        _save_raw_upload(raw)
        try:
            with log_duration(logger, "emotion_pipeline.analyze_user_audio", component="emotion"):
                result: UserAudioResult = await loop.run_in_executor(
                    container.emotion_executor, container.emotion_pipeline.analyze_user_audio, raw
                )
        except AudioDecodeError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except SherpaModelFilesMissing as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except VADOutputInvalid as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        return EmotionVADResponse(
            transcript=result.transcript,
            asr=ASRInfo(),
            user_vad=UserVAD(
                mode="multimodal",
                valence=result.valence,
                arousal=result.arousal,
                dominance=result.dominance,
            ),
        )

    # Text-only fallback: no audio, so no multimodal (WavLM) branch is
    # possible — never fabricate zero audio/embeddings to force it.
    try:
        with log_duration(logger, "emotion_pipeline.analyze_user_text", component="emotion"):
            result = await loop.run_in_executor(
                container.emotion_executor, container.emotion_pipeline.analyze_user_text, text
            )
    except VADOutputInvalid as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return EmotionVADResponse(
        transcript=result.transcript,
        asr=None,
        user_vad=UserVAD(
            mode="text",
            valence=result.valence,
            arousal=result.arousal,
            dominance=result.dominance,
        ),
    )
