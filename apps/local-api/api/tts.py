import asyncio
import io
import wave

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response, StreamingResponse

from api.dependencies import get_container
from core.container import ServiceContainer
from core.tts_coordinator import TTSInterruptCoordinator
from schemas.tts import TTSRequest
from service.tts_service import TTSService

router = APIRouter(prefix="/v1/audio", tags=["TTS"])

_CHUNK_SIZE = 4096  # bytes per streamed chunk


def _synthesize_to_wav(tts: TTSService, voice_id: str, text: str) -> bytes:
    voice = tts.get_voice(voice_id)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        voice.synthesize_wav(text, wf)
    return buf.getvalue()


def _resolve_voice(tts: TTSService, voice: str) -> str:
    """Resolve a requested voice id, raising a stable error code instead of
    silently falling back to whichever voice happens to load first.
    """
    if voice == "default":
        voices = tts.list_voices()
        if not voices:
            raise HTTPException(
                status_code=503,
                detail={"code": "TTS_NO_VOICES_AVAILABLE", "message": "No Piper voice is installed."},
            )
        return voices[0]["id"]

    if not tts.has_voice(voice):
        raise HTTPException(
            status_code=404,
            detail={"code": "TTS_UNKNOWN_VOICE", "message": f"Unknown voice '{voice}'."},
        )
    return voice


async def _stream_chunks(payload: bytes, coordinator: TTSInterruptCoordinator, my_id: int, chunk_size: int = _CHUNK_SIZE):
    """Yield `payload` in `chunk_size` pieces, stopping early once a newer
    generation has claimed the stream (see `TTSInterruptCoordinator`).
    """
    for offset in range(0, len(payload), chunk_size):
        if coordinator.is_cancelled(my_id):
            return
        yield payload[offset : offset + chunk_size]
        await asyncio.sleep(0)  # yield control so a newer request can be observed


@router.get("/voices")
async def list_voices_endpoint(container: ServiceContainer = Depends(get_container)):
    return {"voices": [v["id"] for v in container.tts.list_voices()]}


@router.post("/speech")
async def synthesize(body: TTSRequest, container: ServiceContainer = Depends(get_container)):
    """WAV hoàn chỉnh — cũng interrupt stream cũ nếu có."""
    if not body.input.strip():
        raise HTTPException(status_code=422, detail={"code": "TTS_EMPTY_INPUT", "message": "Trường 'input' rỗng."})

    # Interrupt any in-progress TTS streams
    container.tts_coordinator.cancel_active()

    voice_id = _resolve_voice(container.tts, body.voice)
    try:
        wav_bytes = _synthesize_to_wav(container.tts, voice_id, body.input)
    except Exception as e:
        raise HTTPException(status_code=500, detail={"code": "TTS_SYNTHESIS_FAILED", "message": str(e)}) from e

    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={"Content-Disposition": "inline; filename=speech.wav"},
    )


@router.post("/speech/stream")
async def synthesize_stream(body: TTSRequest, container: ServiceContainer = Depends(get_container)):
    """Chunked WAV stream — dừng ngay khi có luồng mới hoặc input mới đến."""
    if not body.input.strip():
        raise HTTPException(status_code=422, detail={"code": "TTS_EMPTY_INPUT", "message": "Trường 'input' rỗng."})

    # Claim a new generation slot — invalidates all previous streams
    my_id = container.tts_coordinator.begin_turn()

    voice_id = _resolve_voice(container.tts, body.voice)
    try:
        wav_bytes = _synthesize_to_wav(container.tts, voice_id, body.input)
    except Exception as e:
        raise HTTPException(status_code=500, detail={"code": "TTS_SYNTHESIS_FAILED", "message": str(e)}) from e

    return StreamingResponse(
        _stream_chunks(wav_bytes, container.tts_coordinator, my_id),
        media_type="audio/wav",
        headers={"Content-Disposition": "inline; filename=speech.wav"},
    )
