# Mitsuka — Project Context

> Keep this file concise and update it whenever the architecture changes.

## What this is
Local AI companion stack: speech-to-text, text-to-speech, emotion (V/A/D) analysis, and an
LLM "brain" with RAG memory. `airi/` is the upstream [moeru-ai/airi](https://github.com/moeru-ai/airi)
front-end (Vue monorepo, credited to its authors); everything under `apps/local-api/` and the
`tools/legacy/` Python servers is original work.

## Folders
```
Mitsuka/
├─ airi/                    # Upstream airi monorepo (front-end apps, Tamagotchi UI)
├─ ModelPreVoice/           # Whisper model weights & tokenizer
├─ assets/
│  └─ models/
│     └─ voices/            # Piper TTS voices (*.onnx + *.onnx.json), tracked in git
├─ tools/
│  └─ legacy/               # whisper_server.py, test_transcribe.py — not the production entry point
├─ apps/
│  └─ local-api/            # All local services (single FastAPI app, port 8000)
│     ├─ main.py            # create_app() factory; model/service construction happens in lifespan
│     ├─ api/                # HTTP routes: chat, vad, emotion_vad, tts, whisper
│     ├─ application/        # Cross-endpoint policy (prepare_turn: shared by /v1/chat and /v1/chat/stream)
│     ├─ core/                # App composition: ServiceContainer, TTSInterruptCoordinator,
│     │                        BackgroundTaskRegistry, structured logging, ASGI request-context middleware
│     ├─ service/             # Sync model services: VADService (text→V/A/D), audio emotion, TTS, whisper
│     ├─ model/                # BERT V/A/D regression model (tanh head → [-1,1]) + checkpoint
│     ├─ brain/                # LLM brain: RAG, memory, emotion state, background fact extraction
│     ├─ schemas/              # Pydantic request/response models, incl. the /v1/chat/stream SSE envelope
│     ├─ scripts/              # Dev utilities (test_rag, backfill_vad, inspect_*) — not part of the test suite
│     ├─ tests/
│     │  ├─ unit/              # Pure functions, no I/O
│     │  ├─ integration/       # Full ASGI stack via httpx, all external deps faked
│     │  └─ smoke/              # Needs a real running server — opt-in only, excluded from pytest.ini
│     └─ data/brain.db         # SQLite conversation + facts store (gitignored, runtime data)
```

## Data flow (chat turn)
1. `POST /v1/chat` (or `/v1/chat/stream`) — [apps/local-api/api/chat.py](apps/local-api/api/chat.py)
2. Both endpoints share one policy via `prepare_turn` —
   [apps/local-api/application/conversation_turn.py](apps/local-api/application/conversation_turn.py):
   `should_use_rag` heuristic (regex, no LLM) decides whether to retrieve; if so, RAG runs
   (embed query → Qdrant search → RRF fusion → context string —
   [apps/local-api/brain/rag_service.py](apps/local-api/brain/rag_service.py)); then `build_messages`
   assembles the system prompt (context + current emotion label) + recent SQLite history —
   [apps/local-api/brain/nodes/generate.py](apps/local-api/brain/nodes/generate.py).
3. `container.llm.chat(...)` (buffered) or `container.llm.stream_chat(...)` (SSE) calls Ollama.
4. Response's V/A/D inferred (`EmotionService.infer`), blended with retrieved memories' V/A/D
   into the current system emotional state; returned to the client (`emotion` + `state` fields on
   `ChatResponse`, or an `emotion` SSE event before `done`).
5. Background (client does not wait, tracked by `ServiceContainer.tasks` — a `BackgroundTaskRegistry`
   that logs failures and drains on shutdown): save turn to SQLite with V/A/D + emotion, index the
   exchange into Qdrant with V/A/D payload, LLM decides whether to extract long-term facts —
   [apps/local-api/brain/background.py](apps/local-api/brain/background.py)

### `/v1/chat/stream` SSE envelope
Every event is `{"type": "delta"|"emotion"|"error"|"done", "turn_id": "...", ...}` — see
[apps/local-api/schemas/chat.py](apps/local-api/schemas/chat.py). `error` events carry a stable
`code` (e.g. `LLM_UNAVAILABLE`) + `message` + `retryable`; the stream still ends with a `done`
event after an error. The frontend parser lives at
`airi/packages/stage-ui/src/composables/local-conversation-sse.ts`.

## Emotion pipeline
- **Model**: BERT regression head with `tanh` → valence/arousal/dominance each in **[-1, 1]**
  ([apps/local-api/model/vad_model.py](apps/local-api/model/vad_model.py), served by `service/vad_service.py`).
- **Mapper**: deterministic nearest-prototype in PAD space over a ~28-emotion taxonomy —
  [apps/local-api/brain/emotion_mapper.py](apps/local-api/brain/emotion_mapper.py). Same V/A/D always → same label.
- **State**: `EmotionService.current_state` = response V/A/D × 0.6 + mean(retrieved memories'
  V/A/D) × 0.4, then re-mapped to a label —
  [apps/local-api/brain/emotion_service.py](apps/local-api/brain/emotion_service.py).
- **Consumption**: agents use the stored **emotion label**, not raw numbers — RAG context lines
  are tagged `(felt: <emotion>)`, and the system prompt carries the last assistant emotion.
- `/emotion-vad` additionally fuses audio (wav2vec2) and text V/A/D at 0.7/0.3, and accepts a
  `language` form field forwarded to Whisper.
- **Backfill**: `python scripts/backfill_vad.py` fills V/A/D + emotion for old assistant rows.

## RAG pipeline
- Embeddings: `paraphrase-multilingual-MiniLM-L12-v2` (384-dim, CPU).
- Vector store: Qdrant, cosine distance; **in-memory by default** (`qdrant_url` empty in
  [apps/local-api/brain/config.py](apps/local-api/brain/config.py)) — indexed conversation memories
  do not survive restarts unless a Qdrant URL is configured. Seed docs via `POST /v1/chat/seed`.
- Memory payload schema: `{content, type: "conversation", response, valence, arousal,
  dominance, emotion, timestamp}`.
- Fusion: reciprocal rank fusion (k=60), top-5.

## DB schema (SQLite, `apps/local-api/data/brain.db`)
```sql
conversations(id, role, content, timestamp,
              valence REAL, arousal REAL, dominance REAL, emotion TEXT)  -- V/A/D nullable; set for assistant rows
facts(id, key UNIQUE, value, embedding_id, updated_at)
```
Schema migrations are additive `ALTER TABLE`s applied in `MemoryService.initialize()`.

## App composition & lifecycle
- `main.create_app(settings)` is a pure factory — importing `main` never touches a real model.
  All model/service construction happens inside the FastAPI `lifespan`, via
  `core.container.ServiceContainer.create(settings)`, and is torn down via `container.shutdown()`.
- Routes depend on `Depends(get_container)` (`api/dependencies.py`) instead of importing services
  directly — tests override this via `app.dependency_overrides` or (more commonly here) by
  monkeypatching `ServiceContainer.create` itself, so no real model/Ollama/Qdrant is ever touched
  in the test suite.
- TTS barge-in/interrupt is owned by `core.tts_coordinator.TTSInterruptCoordinator`
  (`begin_turn`/`is_cancelled`/`cancel_active`) instead of a bare `app.state.active_tts_id` counter.
- `GET /health/live` — process is up, no dependency checks. `GET /health` is kept as an alias
  (the frontend's `Stage.vue` local-server probe depends on the literal `/health` path).
  `GET /health/ready` — required in-process dependencies constructed successfully.

## Logging & correlation
- `core/logging.py`: structured logging (human-readable console in dev, single-line JSON when
  `LOG_JSON=1`), with `request_id`/`turn_id` propagated via `contextvars` so call sites don't
  thread them through function signatures.
- `core/http_middleware.py`'s `RequestContextMiddleware` (plain ASGI, not `BaseHTTPMiddleware` —
  see its docstring for why) assigns/echoes `X-Request-Id` on every response and logs one
  structured line per request.
- `turn_id` is generated per chat turn, included in every SSE event and in `ChatResponse.turn_id`,
  and stays bound (via `bind_turn_id`) across the fire-and-forget background memory task too — a
  turn can be grepped end-to-end from the browser to `brain.background`'s log lines.
- Unexpected exceptions get a global handler returning `{"error": {"code": "INTERNAL_ERROR",
  "message": "...", "request_id": "..."}}` — full traceback stays server-side.

## Conventions
- Python: `from __future__ import annotations`, module-level `logger = logging.getLogger(__name__)`,
  services as plain classes injected via constructors, FastAPI DI via `Depends(get_container)`.
- Sync torch models run off the event loop (`asyncio.to_thread` / `container.emotion_executor`).
- Anything slow after the response goes through `container.tasks.spawn(coro, name=...)`
  (`core/tasks.py`'s `BackgroundTaskRegistry`) — logs failures with traceback, drains on shutdown.
- Config via pydantic-settings (`brain/config.py`), env-overridable, `.env` supported.

## Tests
- `cd apps/local-api && python -m pytest -q` — fully offline, no real model/Ollama/Qdrant needed;
  `tests/conftest.py` monkeypatches `ServiceContainer.create` to build fakes for every service.
- `tests/smoke/test_apis.py` needs a real running server (`python main.py` first) — intentionally
  excluded from `pytest.ini`'s `testpaths`, run directly if needed.

## Important files
| File | Why it matters |
|---|---|
| [apps/local-api/main.py](apps/local-api/main.py) | App factory; composition only, no model loading at import time |
| [apps/local-api/core/container.py](apps/local-api/core/container.py) | Builds/tears down every service once, from the lifespan |
| [apps/local-api/application/conversation_turn.py](apps/local-api/application/conversation_turn.py) | Shared RAG+message-building policy for buffered *and* streaming chat |
| [apps/local-api/schemas/chat.py](apps/local-api/schemas/chat.py) | The `/v1/chat/stream` SSE envelope (versioned, discriminated by `type`) |
| [apps/local-api/brain/emotion_mapper.py](apps/local-api/brain/emotion_mapper.py) | The V/A/D → emotion taxonomy (edit here to tune labels) |
| [apps/local-api/brain/config.py](apps/local-api/brain/config.py) | All tunables (Ollama model, Qdrant, model paths, logging, CORS) |
| [apps/local-api/tests/conftest.py](apps/local-api/tests/conftest.py) | Fake service fixtures — read this before adding a new test |
