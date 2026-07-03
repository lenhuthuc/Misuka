"""
Local Whisper transcription server — OpenAI-compatible /v1/audio/transcriptions endpoint.
Uses faster-whisper for efficient CPU/GPU inference with Vietnamese support.

Install:  pip install faster-whisper fastapi uvicorn python-multipart
Run:      python whisper_server.py
          (defaults to http://localhost:9000)

The app (stage-tamagotchi / stage-web) auto-detects this server and
configures openai-compatible-audio-transcription to use it.
"""

import io
import os
import tempfile

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_SIZE = os.environ.get("WHISPER_MODEL", "small")   # tiny / base / small / medium / large-v3
DEVICE     = os.environ.get("WHISPER_DEVICE", "cpu")    # cpu / cuda
COMPUTE    = os.environ.get("WHISPER_COMPUTE", "int8")  # int8 / float16 / float32
HOST       = os.environ.get("WHISPER_HOST", "0.0.0.0")
PORT       = int(os.environ.get("WHISPER_PORT", "9000"))

print(f"[Whisper] Loading model '{MODEL_SIZE}' on {DEVICE} ({COMPUTE}) …")
model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE)
print(f"[Whisper] Model ready. Listening on http://{HOST}:{PORT}")

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="Whisper Transcription Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_SIZE}


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "whisper-1",
                "object": "model",
                "owned_by": "local",
            },
        ],
    }


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file:     UploadFile = File(...),
    model:    str        = Form("whisper-1"),
    language: str        = Form(None),    # e.g. "en" for English
    prompt:   str        = Form(None),
    response_format: str = Form("json"),
):
    audio_bytes = await file.read()

    # Write to a temp file because faster-whisper expects a path or file-like object
    suffix = os.path.splitext(file.filename or ".webm")[1] or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        segments, info = model.transcribe(
            tmp_path,
            language=language or "en",      # default to English
            initial_prompt=prompt or None,
            vad_filter=True,                # skip silence
            vad_parameters={"min_silence_duration_ms": 300},
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        os.unlink(tmp_path)

    if response_format == "text":
        return text

    return {
        "text": text,
        "language": info.language,
        "duration": info.duration,
    }


if __name__ == "__main__":
    uvicorn.run("whisper_server:app", host=HOST, port=PORT, reload=False)
