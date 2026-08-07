import os
import tempfile

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from api.dependencies import get_container
from core.container import ServiceContainer

router = APIRouter(prefix="/v1/audio", tags=["Whisper STT"])


@router.post("/transcriptions")
async def transcriptions(
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    language: str | None = Form("en"),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
    prompt: str | None = Form(None),
    container: ServiceContainer = Depends(get_container),
):
    """OpenAI-compatible transcription endpoint.
    AIRI's openai-compatible-audio-transcription provider posts here.
    """
    suffix = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        text = container.whisper.transcribe(tmp_path, language=language)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        os.unlink(tmp_path)

    if response_format == "text":
        return text

    # json / verbose_json / srt / vtt — all return {"text": ...} for AIRI
    return JSONResponse({"text": text})
