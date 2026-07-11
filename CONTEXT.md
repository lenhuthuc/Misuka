# Mitsuka — Project Context

> Keep this file concise and update it whenever the architecture changes.

## What this is
Local AI companion stack: speech-to-text, text-to-speech, emotion (V/A/D) analysis, and an
LLM "brain" with RAG memory. `airi/` is the upstream [moeru-ai/airi](https://github.com/moeru-ai/airi)
front-end (Vue monorepo, credited to its authors); everything under `VAD/` and the root Python
servers is original work.

## Folders
```
Mitsuka/
├─ airi/              # Upstream airi monorepo (front-end apps, Tamagotchi UI)
├─ ModelPreVoice/     # Whisper model weights & tokenizer
├─ VAD/               # All local services (single FastAPI app, port 8000)
│  ├─ main.py         # App entry: loads VAD model, mounts routers, lifespan wires brain
│  ├─ api/            # HTTP routes: chat, vad, emotion_vad, tts, whisper
│  ├─ service/        # Sync model services: VADService (text→V/A/D), audio emotion, TTS, whisper
│  ├─ model/          # BERT V/A/D regression model (tanh head → [-1,1]) + checkpoint
│  ├─ brain/          # LLM brain: LangGraph pipeline, RAG, memory, emotion state
│  │  └─ nodes/       # Graph nodes: should_rag, retrieve, generate, decide, memory
│  ├─ schemas/        # Pydantic request/response models
│  ├─ scripts/        # Dev utilities (test_rag, backfill_vad, inspect_*)
│  ├─ tests/          # HTTP test suite (server must be running)
│  └─ data/brain.db   # SQLite conversation + facts store
└─ whisper_server.py  # Standalone Whisper server (port 9000, OpenAI-compatible)
```

## Data flow (chat turn)
1. `POST /v1/chat` (or `/v1/chat/stream`) — [VAD/api/chat.py](VAD/api/chat.py)
2. `should_rag` heuristic (regex, no LLM) decides whether to retrieve — [VAD/brain/nodes/should_rag.py](VAD/brain/nodes/should_rag.py)
3. RAG: embed query → Qdrant search → RRF fusion → context string — [VAD/brain/rag_service.py](VAD/brain/rag_service.py)
4. `generate`: system prompt (context + current emotion label) + recent SQLite history → Ollama — [VAD/brain/nodes/generate.py](VAD/brain/nodes/generate.py)
5. Response's V/A/D inferred (`EmotionService.infer`), blended with retrieved memories' V/A/D
   into the current system emotional state; returned to the client (`emotion` + `state` fields,
   or a final SSE event before `[DONE]`).
6. Background (client does not wait): save turn to SQLite with V/A/D + emotion, index the
   exchange into Qdrant with V/A/D payload, LLM decides whether to extract long-term facts —
   [VAD/brain/background.py](VAD/brain/background.py)

## Emotion pipeline
- **Model**: BERT regression head with `tanh` → valence/arousal/dominance each in **[-1, 1]**
  ([VAD/model/vad_model.py](VAD/model/vad_model.py), served by `service/vad_service.py`).
- **Mapper**: deterministic nearest-prototype in PAD space over a ~28-emotion taxonomy —
  [VAD/brain/emotion_mapper.py](VAD/brain/emotion_mapper.py). Same V/A/D always → same label.
- **State**: `EmotionService.current_state` = response V/A/D × 0.6 + mean(retrieved memories'
  V/A/D) × 0.4, then re-mapped to a label — [VAD/brain/emotion_service.py](VAD/brain/emotion_service.py).
- **Consumption**: agents use the stored **emotion label**, not raw numbers — RAG context lines
  are tagged `(felt: <emotion>)`, and the system prompt carries the last assistant emotion.
- `/emotion-vad` additionally fuses audio (wav2vec2) and text V/A/D at 0.7/0.3.
- **Backfill**: `python scripts/backfill_vad.py` fills V/A/D + emotion for old assistant rows.

## RAG pipeline
- Embeddings: `paraphrase-multilingual-MiniLM-L12-v2` (384-dim, CPU).
- Vector store: Qdrant, cosine distance; **in-memory by default** (`qdrant_url` empty in
  [VAD/brain/config.py](VAD/brain/config.py)) — indexed conversation memories do not survive restarts unless a
  Qdrant URL is configured. Seed docs via `POST /v1/chat/seed`.
- Memory payload schema: `{content, type: "conversation", response, valence, arousal,
  dominance, emotion, timestamp}`.
- Fusion: reciprocal rank fusion (k=60), top-5.

## DB schema (SQLite, `VAD/data/brain.db`)
```sql
conversations(id, role, content, timestamp,
              valence REAL, arousal REAL, dominance REAL, emotion TEXT)  -- V/A/D nullable; set for assistant rows
facts(id, key UNIQUE, value, embedding_id, updated_at)
```
Schema migrations are additive `ALTER TABLE`s applied in `MemoryService.initialize()`.

## Conventions
- Python: `from __future__ import annotations`, module-level `logger = logging.getLogger(__name__)`,
  services as plain classes injected via constructors, FastAPI DI overridden in `main.py`.
- Graph nodes are factories (`make_*_node(deps)`) returning async callables over `BrainState`
  ([VAD/brain/state.py](VAD/brain/state.py)).
- Sync torch models run off the event loop (`asyncio.to_thread` / thread-pool executors).
- Anything slow after the response goes into a fire-and-forget `asyncio.create_task` with a
  done-callback that logs exceptions.
- Config via pydantic-settings (`brain/config.py`), env-overridable, `.env` supported.

## Important files
| File | Why it matters |
|---|---|
| [VAD/main.py](VAD/main.py) | Single entry point; wires every service into `app.state` |
| [VAD/brain/app.py](VAD/brain/app.py) | Brain service factory (`create_services`) |
| [VAD/brain/graph.py](VAD/brain/graph.py) | LangGraph wiring (fast graph in prod; full graph kept for reference) |
| [VAD/brain/emotion_mapper.py](VAD/brain/emotion_mapper.py) | The V/A/D → emotion taxonomy (edit here to tune labels) |
| [VAD/brain/config.py](VAD/brain/config.py) | All tunables (Ollama model, Qdrant, limits) |
| [VAD/tests/test_apis.py](VAD/tests/test_apis.py) | HTTP smoke tests (`python tests/test_apis.py` with server up) |
