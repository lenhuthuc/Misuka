import asyncio

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from api.dependencies import get_container
from core.container import ServiceContainer
from service.audio_preprocessing import AudioDecodeError, prepare_mono_16k

router = APIRouter(prefix="/v1/audio", tags=["Sherpa-ONNX STT"])


@router.post("/transcriptions")
async def transcriptions(
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    response_format: str = Form("json"),
    container: ServiceContainer = Depends(get_container),
):
    """OpenAI-compatible transcription endpoint (Vietnamese-only, via
    Sherpa-ONNX). AIRI's openai-compatible-audio-transcription provider posts
    here — `model`/`response_format` stay for wire compatibility with that
    client; Whisper-only params (`language`, `prompt`, `temperature`) are gone,
    since the configured Sherpa model has no language/task switch.
    """
    raw = await file.read()
    try:
        samples = prepare_mono_16k(raw)
    except AudioDecodeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        # Sherpa's decode is synchronous and CPU-heavy. Running it on the
        # request event loop freezes health checks and every other API route.
        loop = asyncio.get_running_loop()
        text = await loop.run_in_executor(container.emotion_executor, container.asr.transcribe, samples)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if response_format == "text":
        return text

    # json / verbose_json / srt / vtt — all return {"text": ...} for AIRI
    return JSONResponse({"text": text})
