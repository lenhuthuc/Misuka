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
│     ├─ voices/            # Piper TTS voices (*.onnx + *.onnx.json), tracked in git
│     └─ models/            # Live2D Cubism 4 source model (TiredGirl_V1.*) — see "Avatar"
├─ tools/
│  └─ legacy/               # whisper_server.py, test_transcribe.py — not the production entry point
├─ apps/
│  └─ local-api/            # All local services (single FastAPI app, port 8010)
│     ├─ main.py            # create_app() factory; model/service construction happens in lifespan
│     ├─ api/                # HTTP routes: chat, vad, emotion_vad, tts, whisper (Sherpa-ONNX-backed), vision
│     ├─ application/        # Cross-endpoint policy (prepare_turn: shared by /v1/chat and /v1/chat/stream)
│     ├─ core/                # App composition: ServiceContainer, BackgroundTaskRegistry,
│     │                        structured logging, ASGI request-context middleware
│     ├─ service/             # Sync model services: sherpa_asr, text_vad, multimodal_vad,
│     │                        emotion_pipeline (orchestrator), audio_preprocessing,
│     │                        TTS (Piper) + prosody (Fujisaki F0 model)
│     ├─ model/                # PhoBERT/WavLM V/A/D model classes (text_vad.py, multimodal_vad.py,
│     │                        encoders.py) + trained checkpoints + vendored PhoBERT tokenizer
│     ├─ brain/                # LLM brain: RAG, memory, emotion state, background fact extraction
│     ├─ schemas/              # Pydantic request/response models, incl. the /v1/chat/stream SSE envelope
│     ├─ scripts/              # Dev utilities (test_rag, backfill_vad, inspect_*) — not part of the test suite
│     ├─ tests/
│     │  ├─ unit/              # Pure functions, no I/O
│     │  ├─ integration/       # Full ASGI stack via httpx, all external deps faked
│     │  └─ smoke/              # Needs a real running server — opt-in only, excluded from pytest.ini
│     └─ data/brain.db         # SQLite conversation store (gitignored, runtime data)
```

## Data flow (chat turn)
1. `POST /v1/chat` (or `/v1/chat/stream`) — [apps/local-api/api/chat.py](apps/local-api/api/chat.py)
2. Both endpoints share one policy via `prepare_turn` —
   [apps/local-api/application/conversation_turn.py](apps/local-api/application/conversation_turn.py):
   `should_use_rag` heuristic (regex, no LLM) decides whether to retrieve; if so, RAG runs
   (embed query → Qdrant search → RRF fusion → context string —
   [apps/local-api/brain/rag_service.py](apps/local-api/brain/rag_service.py)); then `build_messages`
   assembles the system prompt + recent SQLite history —
   [apps/local-api/brain/nodes/generate.py](apps/local-api/brain/nodes/generate.py).
   The prompt is a **spoken-companion** prompt, not a retrieval-QA one. Its instructions are in
   English (a 1.5B Qwen follows English rules markedly better — a fully Vietnamese prompt made it
   answer "được" to an arithmetic question) with an explicit Vietnamese *output* rule, and it
   states that the RAG block is optional background. The previous "use the provided context
   to answer… if the context does not contain enough information, say so honestly" fired on every
   unrelated question: asked what one plus one was, the model reported it could not determine the
   answer from what it had been given.
3. `container.llm.chat(...)` (buffered) or `container.llm.stream_chat(...)` (SSE) calls Ollama.
4. Response's V/A/D inferred (`EmotionService.infer`), blended with retrieved memories' V/A/D
   into the current system emotional state; returned to the client (`emotion` + `state` fields on
   `ChatResponse`, or an `emotion` SSE event before `done`).
5. Background (client does not wait, tracked by `ServiceContainer.tasks` — a `BackgroundTaskRegistry`
   that logs failures and drains on shutdown): save turn to SQLite with V/A/D + emotion, index the
   exchange into Qdrant with V/A/D payload, enqueue the exchange for fact extraction —
   [apps/local-api/brain/background.py](apps/local-api/brain/background.py). Nothing here calls the
   LLM; see "Background LLM work" below for why.

## Background LLM work
Ollama serves one request at a time per model and decode is memory-bandwidth-bound here, so any
background generation either delays the next turn or halves total throughput. Two pieces enforce
"the user's turn owns the runner":
- **`core/llm_priority.py`'s `LLMPriorityGate`** — chat endpoints wrap their turn in
  `foreground()`; background callers go through `run_when_idle()`, which abandons work the moment
  the conversation stirs. Crucially the gate tracks the *conversation*, not just the generation:
  the `foreground()` block exits at the last generated token, while the reply is still unspoken.
  So `/emotion-vad` reports `mark_active()` (the user is talking now) and `/v1/audio/speech`
  reports `hold_active(<clip duration>)` (that much reply is about to play), and background work
  additionally waits `llm_quiet_seconds` past all of it. Getting this wrong is not just a queued
  request — a background prompt evicts the chat model's cached prefix in Ollama, so work that is
  cancelled the instant the user speaks still bills ~1s of re-prefill to the turn it interrupted.
  Because the reply is spoken a sentence at a time, `hold_active` only ever covers the clips
  already asked for, so the reservation runs dry in *every* inter-sentence gap. `wait_until_spoken`
  therefore also requires a lull of `llm_speech_lull_seconds` with no new clip before it will
  assume the reply ended — without it the exchange embedding was released by the gap between two
  sentences and then widened that same gap by competing with the render inside it.
- The gate once carried a wider invariant — background *LLM* calls must never overlap a turn — for
  `brain/curator.py`'s `MemoryCurator`, which mined finished exchanges for durable facts on the same
  Ollama runner a chat turn needs. That component, the `facts` table it wrote, and the
  `run_when_idle`/quiet-window machinery it waited on were all removed: nothing on the server issues
  background LLM calls any more. Durable memory past the history window is the vector store's job.
  If a background LLM caller is ever reintroduced, that gate has to come back with it.

### `/v1/chat/stream` SSE envelope
Every event is `{"type": "delta"|"emotion"|"error"|"done", "turn_id": "...", ...}` — see
[apps/local-api/schemas/chat.py](apps/local-api/schemas/chat.py). `error` events carry a stable
`code` (e.g. `LLM_UNAVAILABLE`) + `message` + `retryable`; the stream still ends with a `done`
event after an error. The frontend parser lives at
`airi/packages/stage-ui/src/composables/local-conversation-sse.ts`.

## Speech-to-text (Sherpa-ONNX)
Whisper (`faster-whisper`) has been fully replaced by **Sherpa-ONNX** — Vietnamese-only, no
`language`/`task` switch. `service/sherpa_asr_service.py` loads one `sherpa_onnx.OfflineRecognizer`
(transducer, greedy search, CPU/int8) at startup via `ServiceContainer.create`; it is never
recreated per-request. Model files are large binaries, not committed — download the default model:
```sh
pip install "huggingface_hub[cli]"
huggingface-cli download csukuangfj2/sherpa-onnx-zipformer-vi-30M-int8-2026-02-09 \
  --local-dir assets/models/sherpa-onnx-zipformer-vi-30M-int8-2026-02-09
```
Override individual files via `SHERPA_ONNX_{TOKENS,ENCODER,DECODER,JOINER}` (or the whole
directory via `SHERPA_ONNX_MODEL_DIR`) — see `apps/local-api/.env.example`. Missing files raise a
clear `SherpaModelFilesMissing` at startup rather than failing silently.

`POST /v1/audio/transcriptions` (OpenAI-compatible, used by AIRI) and `POST /emotion-vad` both go
through it. ASR always sees the **full** uploaded audio — never trimmed to a fixed duration.

**Transcripts are case-folded before they leave the service** (`normalize_transcript`). This
model's entire BPE vocabulary is uppercase, so every raw transcript SHOUTS, and that is a
correctness bug rather than a cosmetic one: measured on `qwen2.5:1.5b` *and* `qwen2.5:3b`, the
transcript "ĐỐ BẠN LÀ MỘT CỘNG MỘT BẰNG BAO NHIÊU" was answered as a question about the
assistant's identity by both models, while the same sentence lowercased was answered "một cộng
một bằng hai" by both — four of four caps runs wrong, four of four lowercase runs right. PhoBERT
(`vinai/phobert-base-v2`) is likewise trained on lowercase-dominant text, so the V/A/D heads were
seeing out-of-distribution input too.

## Text-to-speech (Piper + Fujisaki prosody)
`POST /v1/audio/speech` renders **one complete utterance per request**, and an "utterance" is
whatever the caller asks for: the local-conversation client asks for **one sentence at a time**,
as each boundary arrives out of the token stream, so speech starts before generation finishes.
There is still no chunked *route* — `/v1/audio/speech/stream` and `core/tts_coordinator.py` are
gone; the pipelining lives entirely in the client's two chains
(`local-conversation-tts.ts`: synthesis one chunk ahead of a serial playback queue).
The two failures that killed the earlier sentence-by-sentence attempt are fixed, not avoided:
- The frontend's `extractSentences` only closed a sentence on a terminator *followed by
  whitespace*, so a reply ending in "." left its whole text in the tail buffer — which was then
  spoken once as the tail and once more as the "no sentence was ever queued" fallback. Every
  single-sentence answer was spoken twice, the second time from the top. `extractSpeakableChunks`
  now returns the unconsumed remainder rather than a flag, so every character leaves exactly once.
- A pitch contour is planned across a clause, and a per-sentence request had no reply-level V/A/D
  to send. `auto_prosody: true` makes the route read *that sentence's* emotion off its own words —
  and a sentence is a clause, which is the span the Fujisaki contour wants anyway.

Per-sentence timing is logged (`operation=tts_auto_prosody` / `tts_render` with `duration_ms`),
because the silence a user hears between two sentences is one of those two calls plus whatever
else is on the cores.

The optional `valence`/`arousal`/`dominance` fields (the reply's own `agent_vad`, in **[0, 1]**)
drive `service/prosody.py`. Omit them and the voice renders exactly as it did before prosody
existed; `TTS_PROSODY_DEPTH=0` disables the stage globally.

Three effects, applied by deliberately different means:
| Effect | How | Why that way |
|---|---|---|
| Tempo | Piper `length_scale` + inter-sentence pause | free, no post-processing |
| Base pitch | Piper renders `length_scale * pitch_scale` long, then polyphase-resampled back down | pure resampling invents nothing, so nothing can sound synthetic |
| Contour | Fujisaki phrase + accent commands → per-sample ratio → SOLA granular shift | only the mean-removed residual (a few percent) goes through the shifter, where it is transparent |

The Fujisaki model (Fujisaki & Hirose, 1984) generates log-F0 as a baseline plus a phrase command
per clause (critically-damped impulse response — the declination) and an accent command per
2-syllable group (step response — local prominence). Emotion moves the command *amplitudes* only,
never the filter constants, so every setting stays the same speaker. Amplitudes are far below
literature values because this contour is added *on top of* the F0 Piper already produces.
Question sentences get a boosted final accent whose step-down falls past the end of the audio, so
the contour is still rising when the utterance stops.

`pitch_shift_variable`'s grain size is measured, not guessed: at ratio 1.06 a 1024-sample grain
retains 88% of source RMS (grains drift out of phase *within* the window and cancel), 512 retains
96%, and the 256/128 default retains ~98.5% with measured periodicity slightly above the source's.
Measured end to end on `fusion_E_ling75_acoustic100`: neutral V/A/D is within 1% of the untouched
render, high arousal comes out ~14% faster and +2.2 semitones, low arousal ~10% slower and −1.3
semitones with a narrower F0 spread. RTF stays under 0.08.

## Emotion pipeline
Two independently-trained checkpoints, two separate PhoBERT instances (never shared — see
`model/encoders.py`'s `PhoBertEncoder` docstring). Both output **[0, 1]** natively (their heads have
no final activation); scores are clamped and checked for NaN/Inf in
`service/text_vad_service.py`/`service/multimodal_vad_service.py`.
- **`user_vad`** (`model/multimodal_vad.py`, `model/best_multimodal_vad.pt`): WavLM
  (`microsoft/wavlm-base-plus` architecture) on a center-cropped/padded 4s@16kHz window of the raw
  audio + PhoBERT (`vinai/phobert-base-v2` architecture, word-segmented via `underthesea`) on the
  Sherpa transcript of the *full* audio, concatenated and regressed. `mode: "multimodal"`. A
  text-only user turn (no audio) falls back to the text-only model instead (`mode: "text"`) — see
  `service/emotion_pipeline.py`'s `analyze_user_text`; no audio is ever fabricated to force the
  multimodal path.
- **`agent_vad`** (`model/text_vad.py`, `model/best_text_vad.pt`): PhoBERT-only, on the agent's
  complete response text (accumulated fully before inference — never per-token). Computed once per
  turn in `api/chat.py`'s `_agent_vad`, sent in `ChatResponse.agent_vad` (buffered) or in the
  stream's final `done` event (`ChatStreamDoneEvent.agent_vad`). Never touches user audio,
  transcript, `user_vad`, or WavLM embeddings.
- **Legacy signed [-1, 1] convention** — `response_policy.py`, the Live2D driver
  (`airi/packages/stage-ui-live2d/.../emotion-vad.ts`), `EmotionService.current_state`'s memory
  blend, and `/vad`'s response all still expect V/A/D in **[-1, 1]**. `TextVADService.predict()` /
  `.predict_signed()` rescale the [0, 1] model output (`x*2-1`) for exactly these callers — the
  [0, 1] range only appears in the newer `user_vad`/`agent_vad` fields.
- **Mapper**: deterministic nearest-prototype in PAD space over a ~28-emotion taxonomy, expects
  signed [-1, 1] input — [apps/local-api/brain/emotion_mapper.py](apps/local-api/brain/emotion_mapper.py).
- **Consumption**: agents use the stored **emotion label**, not raw numbers — RAG context lines
  are tagged `(felt: <emotion>)`, and the system prompt carries the last assistant emotion.
- **Backfill**: `python scripts/backfill_vad.py` fills V/A/D + emotion for old assistant rows.

## Avatar (Live2D)
- **Model**: `TiredGirl_V1`, shipped as the single preset `preset-live2d-1` from
  `airi/packages/stage-ui/src/assets/live2d/preset/tiredgirl.zip` (committed — unlike
  `assets/live2d/models/*`, which upstream gitignores and downloads at build time). Rebuild it
  from `assets/models/models/` with
  `node airi/packages/stage-ui-live2d/scripts/pack-live2d-preset.mjs`.
- The model ships **no motions and no expressions**, so the emotion-motion map
  (`constants/emotions.ts`) has nothing to play — every expression is parameter-driven instead.
- **Cubism Core version is load-bearing**: the Core caps which `moc3` file version it accepts, and
  a too-new model fails only as `CubismMoc.create` throwing `Error: Unknown error`.
  Core 5.0.0 (SDK 5-r.3, what upstream's `DownloadLive2DSDK` pins) tops out at moc3 v5;
  `TiredGirl_V1.moc3` is v6 (Cubism Editor 5.1) and needs Core 6.0.1 (SDK 5-r.5). That's why the
  apps use `airi/packages/stage-ui-live2d/vite/download-cubism-core.mjs` instead — bump
  `CUBISM_SDK_VERSION` there *and* the `<script src>` in each app's `index.html` together.
- Core 6 is one API break ahead of the Cubism **4** framework bundled in
  `pixi-live2d-display@0.4.0`: `renderOrders` moved from `model.drawables` to the model root, and
  the framework's read of the old location blows up on the first rendered frame
  (`doDrawModel` → `Cannot read properties of undefined (reading '0')`).
  `src/utils/live2d-core-compat.ts` re-exposes it; everything else the framework touches is
  unchanged in Core 6. Delete that shim when pixi-live2d-display ships a Cubism 5 framework.
- **V/A/D → parameters**: `airi/packages/stage-ui-live2d/src/composables/live2d/emotion-vad.ts`.
  V/A/D is first combined into affect terms (`joy`, `anger`, `sorrow`, `energy`) — the same
  negative valence reads as anger when dominant and sadness when submissive, and those want
  opposite brows — then written to brows/eyes/mouth-form/cheek/posture every frame.
  `useEmotionStore().emotion` reaches it as the `emotionVad` prop:
  `Stage.vue` → `Live2D.vue` → `live2d/Model.vue`. Toggle/scale via
  `settings/live2d/emotion-vad-{enabled,intensity}`.
- **Frame plugin order** (all `final`, in `Model.vue`): expression → auto-blink → emotion →
  lip sync. Emotion scales the eyes *multiplicatively* so a blink still closes them, and writes
  the resting `ParamMouthOpenY` that lip sync releases to.
- **Mouth open/close**: `mouthOpenSize`/`nowSpeaking` on `useSpeakingStore`. Local mode plays
  Piper audio through the Web Audio graph (`local-conversation-tts.ts`) and taps an
  `AnalyserNode` for the opening; the mouth closes over a 200 ms release once a sentence ends.
  `mouthOpenSource` arbitrates between that and the cloud speech pipeline's rAF loop.

## Vision captioning (chat image attachments)
The chat model (Qwen, via Ollama) is text-only and never receives pixels — an image attached in
the Mitsuka UI reaches it only as words. `POST /v1/vision/caption`
([apps/local-api/api/vision.py](apps/local-api/api/vision.py)) hands the upload to
`CaptionService` ([apps/local-api/brain/caption_service.py](apps/local-api/brain/caption_service.py)),
a thin adapter over the `vision/` pipeline ([apps/local-api/vision/README.md](apps/local-api/vision/README.md)):
YOLO11n + RapidOCR + CLIP run over the image and its Vietnamese `fast_summary` — the objects
found and the text read — is what comes back as the caption. Local ONNX graphs from
`assets/models/vision/`, nothing downloaded at runtime. The image also stays in the pipeline's
per-session buffer, so `VisionPipeline.answer()` could later answer questions about it without a
re-upload; the chat path does not use that yet. The frontend (`apps/stage-web/src/pages/index.vue`'s `send()`)
calls this first when a message has an attached image, folds the returned description into the
text sent to `/v1/chat/stream`, but keeps the *displayed* chat bubble to just what the user typed
(or nothing) plus the image itself — the auto-generated description is context for Mitsuka, never
something shown back to the user as if they'd written it. Best-effort like RAG/web search: a
disabled (`vision_captioning_enabled=false`), unavailable, or timed-out (`vision_caption_timeout_seconds`,
default 30s) captioner degrades to `caption=""` rather than erroring, and the frontend falls back to
sending the typed text alone (or, with no typed text either, just shows the image locally with no
AI turn at all — nothing for a text-only model to answer).

## RAG pipeline
- Embeddings: `paraphrase-multilingual-MiniLM-L12-v2` (384-dim, CPU).
- Vector store: Qdrant, cosine distance; **in-memory by default** (`qdrant_url` empty in
  [apps/local-api/brain/config.py](apps/local-api/brain/config.py)) — indexed conversation memories
  do not survive restarts unless a Qdrant URL is configured. Seed docs via `POST /v1/chat/seed`.
- Memory payload schema: `{content, type: "conversation", response, valence, arousal,
  dominance, emotion, timestamp}`.
- Fusion: reciprocal rank fusion (k=60), top-5.
- **Two gates, for two different questions.** `brain/nodes/should_rag.py`'s `should_use_rag`
  (regex, no LLM) answers "can this *query* benefit from retrieval at all" — short utterances,
  greetings, and questions about the conversation itself skip the search entirely. It cannot know
  whether the store holds anything relevant, and Qdrant returns its `top_k` nearest points however
  far away they are, so `rag_min_score` is a cosine floor applied to the **raw hits, before RRF**
  (fusion scores by rank, which makes "best of a bad lot" indistinguishable from "relevant").
  `RAG | raw_hits=N above_floor=N after_rrf=N` in the log shows where hits are lost.

## DB schema (SQLite, `apps/local-api/data/brain.db`)
```sql
conversations(id, role, content, timestamp,
              valence REAL, arousal REAL, dominance REAL, emotion TEXT)  -- V/A/D nullable; set for assistant rows
```
A `facts` table and a `pending_curation` queue used to live here for the memory curator. Both are
gone from the DDL; a database created before the removal still carries them, unread — drop them with
`DROP TABLE IF EXISTS facts; DROP TABLE IF EXISTS pending_curation;` if you want the file tidy.
Schema migrations are additive `ALTER TABLE`s applied in `MemoryService.initialize()`.

## App composition & lifecycle
- `main.create_app(settings)` is a pure factory — importing `main` never touches a real model.
  All model/service construction happens inside the FastAPI `lifespan`, via
  `core.container.ServiceContainer.create(settings)`, and is torn down via `container.shutdown()`.
- Routes depend on `Depends(get_container)` (`api/dependencies.py`) instead of importing services
  directly — tests override this via `app.dependency_overrides` or (more commonly here) by
  monkeypatching `ServiceContainer.create` itself, so no real model/Ollama/Qdrant is ever touched
  in the test suite.
- TTS barge-in is entirely client-side: one request renders one complete utterance, so there is
  no server-side stream left to cancel. `useLocalConversation` aborts the fetch and stops playback.
  (`core/tts_coordinator.py` and the middleware's `interrupt_paths` existed only for the removed
  chunked route.)
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

## Running (CPU)
```sh
cd apps/local-api
pip install -r requirements.txt
# download the Sherpa-ONNX model first — see "Speech-to-text (Sherpa-ONNX)" above
python main.py   # http://127.0.0.1:8010 — Sherpa always runs on CPU/int8;
                  # the two VAD models auto-select CUDA if available, else CPU
```

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
| [apps/local-api/service/emotion_pipeline.py](apps/local-api/service/emotion_pipeline.py) | Orchestrates ASR + multimodal VAD for user audio; agent-response VAD entry point |
| [apps/local-api/service/prosody.py](apps/local-api/service/prosody.py) | V/A/D → tempo/pitch/Fujisaki contour, plus the SOLA pitch shifter (tune the voice's acting here) |
| [apps/local-api/brain/nodes/generate.py](apps/local-api/brain/nodes/generate.py) | The system prompt — read the comment above it before editing |
| [apps/local-api/model/encoders.py](apps/local-api/model/encoders.py) | Shared PhoBERT/WavLM building blocks, verified against the checkpoints' state_dicts |
| [apps/local-api/brain/config.py](apps/local-api/brain/config.py) | All tunables (Ollama model, Qdrant, model paths, logging, CORS) |
| [apps/local-api/core/llm_priority.py](apps/local-api/core/llm_priority.py) | Keeps background LLM work off the runner for the whole user turn, playback included |
| [apps/local-api/tests/conftest.py](apps/local-api/tests/conftest.py) | Fake service fixtures — read this before adding a new test |
| [apps/local-api/brain/caption_service.py](apps/local-api/brain/caption_service.py) | Image → Vietnamese text via the `vision/` pipeline — how an attached image reaches the text-only chat model |
| [airi/packages/stage-ui/src/composables/local-conversation.ts](airi/packages/stage-ui/src/composables/local-conversation.ts) | The turn pipeline: STT → SSE chat → one spoken utterance |
| [airi/packages/stage-ui-live2d/src/composables/live2d/emotion-vad.ts](airi/packages/stage-ui-live2d/src/composables/live2d/emotion-vad.ts) | V/A/D → Live2D parameter mapping (edit here to tune the avatar's acting) |
| [airi/packages/stage-ui-live2d/src/composables/live2d/motion-manager.ts](airi/packages/stage-ui-live2d/src/composables/live2d/motion-manager.ts) | Per-frame plugin pipeline: blink, emotion, lip sync |
