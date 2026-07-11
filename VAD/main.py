import logging
import os
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# ── CPU thread pinning — phải set trước khi import numpy/onnxruntime/torch ──
def _physical_cores() -> int:
    try:
        out = subprocess.check_output(
            ["wmic", "cpu", "get", "NumberOfCores", "/value"],
            text=True, stderr=subprocess.DEVNULL,
        )
        cores = sum(int(l.split("=")[1]) for l in out.splitlines() if l.startswith("NumberOfCores="))
        if cores > 0:
            return cores
    except Exception:
        pass
    return os.cpu_count() or 1

_CORES = _physical_cores()
os.environ.setdefault("OMP_NUM_THREADS", str(_CORES))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(_CORES))
os.environ.setdefault("MKL_NUM_THREADS", str(_CORES))

# ── Package path (VAD/ root) ─────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from model.vad_model import load_model
from api.vad import get_service, router as vad_router
from api.emotion_vad import get_audio_service, get_vad_service as get_emotion_vad_service, router as emotion_vad_router
from api.tts import router as tts_router
from api.whisper import router as whisper_router
from api.chat import router as chat_router
from service.audio_emotion_service import AudioEmotionService
from service.vad_service import VADService
import brain.app as brain_app
from brain.emotion_service import EmotionService

logger = logging.getLogger(__name__)

# ── VAD model bootstrap (sync, happens at import time) ────────────────────────
MODEL_PATH = str(
    Path(os.getenv("VAD_MODEL_PATH", "model/vad_bert_final.pt"))
    if Path(os.getenv("VAD_MODEL_PATH", "model/vad_bert_final.pt")).is_absolute()
    else HERE / os.getenv("VAD_MODEL_PATH", "model/vad_bert_final.pt")
)

model, tokenizer = load_model(MODEL_PATH)
_vad_service = VADService(model, tokenizer)
_audio_emotion_service = AudioEmotionService()


# ── FastAPI lifespan ──────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger.info("Starting AIRI Local Services...")

    app.state.active_tts_id = 0  # incremented on every new input → interrupts TTS streams

    llm, memory, vector, rag, fast_graph = await brain_app.create_services()
    app.state.brain_graph = fast_graph
    app.state.brain_rag = rag
    app.state.brain_vector = vector
    app.state.brain_llm = llm
    app.state.brain_memory = memory
    app.state.brain_emotion = EmotionService(_vad_service)

    yield

    await brain_app.shutdown_services(llm, memory, vector)
    logger.info("AIRI Local Services stopped")


# ── App factory ───────────────────────────────────────────────────────────────
app = FastAPI(title="AIRI Local Services", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths that carry new user input — arriving here interrupts any TTS stream in progress
_INTERRUPT_PATHS = {
    "/vad",
    "/emotion-vad",
    "/v1/audio/transcriptions",
    "/v1/chat",
}

@app.middleware("http")
async def tts_interrupt_middleware(request: Request, call_next):
    if request.method == "POST" and request.url.path in _INTERRUPT_PATHS:
        request.app.state.active_tts_id += 1
    return await call_next(request)

app.dependency_overrides[get_service] = lambda: _vad_service
app.dependency_overrides[get_emotion_vad_service] = lambda: _vad_service
app.dependency_overrides[get_audio_service] = lambda: _audio_emotion_service

app.include_router(vad_router)
app.include_router(emotion_vad_router)
app.include_router(tts_router)
app.include_router(whisper_router)
app.include_router(chat_router)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": "whisper-1", "object": "model", "owned_by": "local"}],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
