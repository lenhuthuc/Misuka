import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# transformers' threaded state-dict materialization (core_model_loading.py) segfaults
# with an access violation on Windows — force synchronous loading. Unrelated to
# CPU/RAM sizing; this stays regardless of machine specs.
os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")

# ── Package path (apps/local-api root) ────────────────────────────────────────
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402

from api.chat import router as chat_router  # noqa: E402
from api.emotion_vad import router as emotion_vad_router  # noqa: E402
from api.tts import router as tts_router  # noqa: E402
from api.vad import router as vad_router  # noqa: E402
from api.whisper import router as whisper_router  # noqa: E402
from brain.config import Settings, get_settings  # noqa: E402
from core.container import ServiceContainer  # noqa: E402
from core.http_middleware import RequestContextMiddleware  # noqa: E402
from core.logging import configure_logging  # noqa: E402

logger = logging.getLogger(__name__)

# Paths that carry new user input — arriving here interrupts any TTS stream in progress
_INTERRUPT_PATHS = {
    "/vad",
    "/emotion-vad",
    "/v1/audio/transcriptions",
    "/v1/chat",
}


def create_app(settings: Settings | None = None) -> FastAPI:
    """App factory — composition only. Model/service construction happens
    inside `lifespan`, not at import time, so importing this module (or
    overriding `settings` for tests) never touches a real model or backend.
    """
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configure_logging(
            service="mitsuka-api",
            environment=settings.environment,
            json_output=settings.log_json,
            level=settings.log_level,
            log_file=settings.log_file,
        )
        logger.info("Starting AIRI Local Services...")

        app.state.container = await ServiceContainer.create(settings)

        yield

        await app.state.container.shutdown()
        logger.info("AIRI Local Services stopped")

    app = FastAPI(title="AIRI Local Services", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_methods=["*"],
        allow_headers=["*"],
        # Without this, browsers only expose a small safelisted set of
        # response headers to JS — the frontend's `response.headers.get(...)`
        # would silently see `null` for these even though the header is on
        # the wire (see local-conversation-sse.ts's describeHttpFailure).
        expose_headers=["X-Request-Id", "X-Turn-Id"],
    )

    app.add_middleware(RequestContextMiddleware, interrupt_paths=frozenset(_INTERRUPT_PATHS))

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        # Full traceback goes to the server log; the client only ever sees a
        # stable code + the request_id needed to find that log entry.
        # request.state.request_id (not get_request_id()'s contextvar): this
        # handler runs from ServerErrorMiddleware, *outside* RequestContextMiddleware,
        # after its `with bind_request_id(...)` block has already unwound.
        request_id = getattr(request.state, "request_id", None)
        logger.exception("unhandled exception on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "Internal server error.",
                    "request_id": request_id,
                },
            },
        )

    app.include_router(vad_router)
    app.include_router(emotion_vad_router)
    app.include_router(tts_router)
    app.include_router(whisper_router)
    app.include_router(chat_router)

    @app.get("/health/live")
    def health_live():
        """Process is up and serving requests — no dependency checks."""
        return {"status": "ok"}

    @app.get("/health")
    def health_alias():
        """Alias for /health/live, kept for the existing local-server probe in
        `airi/packages/stage-ui/src/components/scenes/Stage.vue`. New callers
        should use /health/live or /health/ready instead.
        """
        return health_live()

    @app.get("/health/ready")
    def health_ready():
        """All required in-process dependencies (VAD, audio-emotion, Whisper,
        Piper) constructed successfully — reachable only once lifespan
        startup has completed, so a failed required dependency means the
        process never gets here at all (startup raises instead).
        """
        return {"status": "ok"}

    @app.get("/v1/models")
    def list_models():
        return {
            "object": "list",
            "data": [{"id": "whisper-1", "object": "model", "owned_by": "local"}],
        }

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
