import asyncio
import io
import logging
import wave

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response, StreamingResponse

from api.dependencies import get_container
from core.container import ServiceContainer
from core.tts_coordinator import TTSInterruptCoordinator
from schemas.tts import TTSRequest
from service.tts_service import TTSService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/audio", tags=["TTS"])

_CHUNK_SIZE = 4096  # bytes per streamed chunk


def wav_duration_seconds(wav_bytes: bytes) -> float:
    """How long `wav_bytes` takes to play, or 0.0 if it cannot be read.

    Use when: reserving conversation time on the LLM priority gate. A reply
    handed to the client is about to occupy the user for exactly this long,
    which is the window background work must stay out of.

    Returns 0.0 rather than raising: a malformed header is a reason to skip the
    reservation, never a reason to fail a synthesis request that succeeded.
    """
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            rate = wf.getframerate()
            return wf.getnframes() / rate if rate else 0.0
    except Exception:
        logger.debug("tts | could not read WAV duration for playback hold", exc_info=True)
        return 0.0


async def _synthesize_to_wav(tts: TTSService, voice_id: str, text: str) -> bytes:
    """Render off the event loop.

    Kokoro takes around two seconds per reply on this hardware; running that
    inline would stall every other request on the loop, including an in-flight
    chat stream.
    """
    return await asyncio.to_thread(tts.synthesize_wav, voice_id, text)


def _resolve_voice(tts: TTSService, voice: str, configured_default: str) -> str:
    """Resolve a requested voice id, raising a stable error code instead of
    silently falling back to whichever voice happens to load first.

    "default" prefers the configured voice and only falls back to registry
    order when that one is not installed — the fallback used to be the whole
    policy, which made the app's voice depend on filesystem ordering.
    """
    if voice == "default":
        voices = tts.list_voices()
        if not voices:
            raise HTTPException(
                status_code=503,
                detail={"code": "TTS_NO_VOICES_AVAILABLE", "message": "No TTS voice is installed."},
            )
        return configured_default if tts.has_voice(configured_default) else voices[0]["id"]

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

    voice_id = _resolve_voice(container.tts, body.voice, container.tts_default_voice)
    try:
        wav_bytes = await _synthesize_to_wav(container.tts, voice_id, body.input)
    except Exception as e:
        raise HTTPException(status_code=500, detail={"code": "TTS_SYNTHESIS_FAILED", "message": str(e)}) from e

    container.llm_gate.hold_active(wav_duration_seconds(wav_bytes))

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

    voice_id = _resolve_voice(container.tts, body.voice, container.tts_default_voice)
    try:
        wav_bytes = await _synthesize_to_wav(container.tts, voice_id, body.input)
    except Exception as e:
        raise HTTPException(status_code=500, detail={"code": "TTS_SYNTHESIS_FAILED", "message": str(e)}) from e

    container.llm_gate.hold_active(wav_duration_seconds(wav_bytes))

    return StreamingResponse(
        _stream_chunks(wav_bytes, container.tts_coordinator, my_id),
        media_type="audio/wav",
        headers={"Content-Disposition": "inline; filename=speech.wav"},
    )
