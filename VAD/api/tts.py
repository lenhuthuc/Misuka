import asyncio
import io
import wave

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from schemas.tts import TTSRequest
from service.tts_service import get_voice, list_voices

router = APIRouter(prefix="/v1/audio", tags=["TTS"])

_CHUNK_SIZE = 4096  # bytes per streamed chunk


def _synthesize_to_wav(voice_id: str, text: str) -> bytes:
    voice = get_voice(voice_id)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        voice.synthesize_wav(text, wf)
    return buf.getvalue()


def _resolve_voice(voice: str) -> str:
    return voice if voice != "default" else next(iter(list_voices()), {}).get("id", "")


@router.get("/voices")
async def list_voices_endpoint():
    return {"voices": [v["id"] for v in list_voices()]}


@router.post("/speech")
async def synthesize(body: TTSRequest, request: Request):
    """WAV hoàn chỉnh — cũng interrupt stream cũ nếu có."""
    if not body.input.strip():
        raise HTTPException(status_code=422, detail="Trường 'input' rỗng.")

    # Interrupt any in-progress TTS streams
    request.app.state.active_tts_id += 1

    try:
        wav_bytes = _synthesize_to_wav(_resolve_voice(body.voice), body.input)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={"Content-Disposition": "inline; filename=speech.wav"},
    )


@router.post("/speech/stream")
async def synthesize_stream(body: TTSRequest, request: Request):
    """Chunked WAV stream — dừng ngay khi có luồng mới hoặc input mới đến."""
    if not body.input.strip():
        raise HTTPException(status_code=422, detail="Trường 'input' rỗng.")

    # Claim a new generation slot — invalidates all previous streams
    request.app.state.active_tts_id += 1
    my_id: int = request.app.state.active_tts_id

    try:
        wav_bytes = _synthesize_to_wav(_resolve_voice(body.voice), body.input)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    app_state = request.app.state

    async def generate():
        for offset in range(0, len(wav_bytes), _CHUNK_SIZE):
            if app_state.active_tts_id != my_id:
                return  # interrupted by newer request
            yield wav_bytes[offset : offset + _CHUNK_SIZE]
            await asyncio.sleep(0)  # yield control so interrupt can be detected

    return StreamingResponse(
        generate(),
        media_type="audio/wav",
        headers={"Content-Disposition": "inline; filename=speech.wav"},
    )
