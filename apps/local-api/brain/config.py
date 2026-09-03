from pathlib import Path
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── Project root ─────────────────────────────────────────────────────────
    base_dir: Path = Field(default=Path(__file__).parent.parent)

    # ── LLM (Ollama) ─────────────────────────────────────────────────────────
    ollama_base_url: str = Field(default="http://localhost:11434")
    # Decode here is memory-bandwidth-bound, not compute-bound: measured on
    # DDR4-3200, throughput scales inversely with model file size and ignores
    # thread count and context size entirely. qwen2.5:3b ran 6.93 tok/s against
    # 13.1 tok/s for 1.5b -- an exact 2x for an exact 2x in weights. That makes
    # file size, not parameter count, the thing to shop on.
    #
    # The stock-model search that picked qwen3:1.7b is over; this is now a
    # fine-tune of it, and the base comparison is kept only because it is what
    # justifies the *size*:
    #   qwen2.5:1.5b  degenerates on any turn needing content.
    #   qwen2.5:3b    answered a Vietnamese turn in Chinese.
    #   qwen3:4b      2.5GB, and reasons aloud in English with thinking off.
    #   gemma3:4b     best Vietnamese of the four, but 3.3GB.
    #   qwen3:1.7b    clean Vietnamese, held the 1-3 sentence rule.
    #
    # `mitsuka-ft` is qwen3:1.7b fine-tuned on this app's own spoken-Vietnamese
    # register, quantised to Q4_K_M (1.1GB, down from the 1.4GB base). Ollama
    # loads it by name from its own registry, so import it once before starting:
    #
    #     cd assets/models/LLM && ollama create mitsuka-ft -f Modelfile
    #
    # The Modelfile is the single source of truth for the persona and for
    # top_p / repeat_penalty / num_ctx. Nothing in this codebase repeats the
    # persona -- see the layout rule at the top of brain/nodes/generate.py,
    # which is what keeps Ollama injecting it. Being a Qwen3 it still reports
    # the `thinking` capability. Chat turns select it adaptively from context
    # confidence; non-chat background generation keeps it disabled.
    ollama_model: str = Field(default="mitsuka-ft")
    # Ollama request options override Modelfile PARAMETERs, so a mismatch here
    # silently wins over `PARAMETER temperature 0.65` in the Modelfile and the
    # fine-tune runs hotter than it was tuned at. Kept equal on purpose; the
    # knobs this file does *not* send (top_p, repeat_penalty, num_ctx) fall
    # through to the Modelfile untouched, which is where they belong.
    llm_temperature: float = Field(default=0.65)
    # A spoken turn that runs past a few sentences costs twice: once to decode
    # now and again on every later turn, since the reply is re-sent inside the
    # history window. 1024 allowed 2,900-character answers that pushed prompts
    # past 11k characters after five turns. This ceiling is also what bounds
    # worst-case time-to-last-token, so it is worth revisiting once the decode
    # rate on this machine is understood: benchmark runs have ranged from 13
    # tok/s down to 2.4 tok/s for the same model with no configuration change,
    # and at the low end 320 tokens is over two minutes of speech.
    llm_max_tokens: int = Field(default=320)

    # ── LLM priority gate ────────────────────────────────────────────────────
    # Only one question is left for the gate to answer -- has the reply been
    # heard yet -- so only the two speech timings remain. The quiet-window
    # setting went with the memory curator, the sole background LLM caller.
    #
    # Backstop for the end-of-speech signal the client sends when its playback
    # queue drains (POST /v1/audio/speech/finished). Exchange indexing waits for
    # that signal so its embedding does not compete with Piper for cores; if the
    # page was closed mid-reply the signal never comes, and after this long the
    # indexing runs regardless rather than dropping the exchange.
    llm_speech_defer_seconds: float = Field(default=90.0)
    # How long the client may go without asking for another clip before the
    # server assumes the reply has finished playing. A reply is spoken one
    # sentence at a time, so the durations reserved by `/v1/audio/speech` only
    # cover the sentences requested so far: without this lull, the reservation
    # runs dry in the gap while the *next* sentence is still rendering, and the
    # exchange embedding starts on top of that render -- widening the very gap
    # it read as the end of the reply. Only has to outlast one sentence's
    # synthesis; the explicit end-of-speech signal short-circuits it.
    llm_speech_lull_seconds: float = Field(default=6.0)

    # ── Embedding model (ONNX / sentence-transformers, CPU-only) ─────────────
    embedding_model_name: str = Field(default="paraphrase-multilingual-MiniLM-L12-v2")
    embedding_dimension: int = Field(default=384)
    embedding_batch_size: int = Field(default=32)

    # ── Qdrant ───────────────────────────────────────────────────────────────
    qdrant_url: str = Field(default="")          # empty → in-memory mode
    qdrant_collection: str = Field(default="brainmaster_docs")
    qdrant_top_k: int = Field(default=5)

    # ── SQLite ───────────────────────────────────────────────────────────────
    sqlite_path: Path = Field(default=Path(__file__).parent.parent / "data" / "brain.db")

    # ── RAG ──────────────────────────────────────────────────────────────────
    rag_num_queries: int = Field(default=1)
    rag_rrf_k: int = Field(default=60)
    # Ceiling on retrieved context, in characters. Prefill is linear in prompt
    # length and dominates time-to-first-token on a CPU runner, so retrieval
    # recall is traded against latency here rather than left unbounded.
    # The two variable context budgets total 5,000 characters by default:
    # 1,500 for retrieval (30%) and 3,500 for recent chat (70%). This makes
    # nearby conversational context the primary source of continuity.
    rag_context_char_budget: int = Field(default=1500)
    # Cosine floor a hit must clear to reach the prompt. Qdrant always returns
    # its `top_k` nearest points, however far away they are, so an unrelated
    # question still retrieved five memories and spent ~900 characters of
    # prefill on them. Fusion ranks hits against each other and cannot tell
    # "best of a bad lot" from "relevant", so the floor has to be applied to
    # the raw similarity, before RRF. 0.0 disables it.
    # 0.50 is deliberately conservative: RAG is optional background and an
    # unrelated memory is more harmful than omitting a weakly related one.
    rag_min_score: float = Field(default=0.50)

    # ── Adaptive reasoning ──────────────────────────────────────────────────
    # Thinking is considered only for substantive turns where RAG was eligible.
    # The strongest signal from long-term RAG, indexed recent history, and the
    # verbatim history window is compared with this threshold.
    reasoning_enabled: bool = Field(default=True)
    reasoning_activation_threshold: float = Field(default=0.50)
    # The fine-tuned chat model reports a thinking capability but does not emit
    # a thinking trace in practice, so a stock Qwen3 performs the optional
    # hidden deliberation pass and Mitsuka still owns the final spoken answer.
    reasoning_model: str = Field(default="qwen3:1.7b")
    # Hard num_predict bounds for that separate pass. A normalized -log curve
    # maps zero confidence to the maximum and decays toward the minimum as the
    # score approaches the threshold; fractional tokens round down.
    reasoning_min_tokens: int = Field(default=64)
    reasoning_max_tokens: int = Field(default=192)
    # Applied after the -log curve and before floor, so it compresses the whole
    # min..max range rather than clipping one end. The hidden pass decodes on
    # the same memory-bandwidth-bound runner as the answer itself, and the note
    # it produces is then prefilled into the answer's own prompt, so every token
    # here is charged twice to time-to-first-token: at 0.80 a zero-confidence
    # turn deliberated for 153 tokens before a word was spoken. 0.45 holds the
    # same confidence-dependent shape at a little over half the cost -- 86
    # tokens at zero confidence, 37 at the midpoint, decaying to a 28-token
    # floor at the threshold -- which still leaves a few lines, and a few lines
    # of hint is all the note is ever used for. Cutting this further trades
    # against note quality; `reasoning_activation_threshold` is the other lever,
    # and it makes the pass fire less often rather than think less each time.
    reasoning_token_scale: float = Field(default=0.45)

    # ── Web search (DuckDuckGo, via `ddgs`) ─────────────────────────────────
    # Fallback for live information neither the frozen local weights nor
    # personal-conversation RAG can have: today's weather, breaking news,
    # current prices. Gated by `decide_web_search` (see
    # brain/nodes/should_search_web.py) so it only fires for the two narrow
    # classes that gate names, not every turn.
    web_search_enabled: bool = Field(default=True)
    # No API key needed -- DDGS scrapes DuckDuckGo directly, in keeping with
    # everything else in this app running local/free rather than against a
    # paid third-party API.
    web_search_max_results: int = Field(default=3)
    # Same reasoning as rag_context_char_budget: prefill is linear in prompt
    # length, so retrieved web snippets are capped rather than sent whole.
    web_search_context_char_budget: int = Field(default=800)
    # DDGS does its own blocking HTTP under the hood; this bounds how long a
    # turn will wait on it before degrading to no web context, same as any
    # other best-effort enrichment in this pipeline.
    web_search_timeout_seconds: float = Field(default=6.0)
    # Biases DuckDuckGo's results toward Vietnamese sources, matching the
    # fine-tune's spoken register. "wt-wt" (DDGS's own default) removes the bias.
    web_search_region: str = Field(default="vn-vi")
    # The third query class, separately switchable because it is the only one
    # that can fire on a turn with no time-sensitive marker in it at all: an
    # ordinary open-world question ("tôm biển là con gì"). Neither the frozen
    # 1.1GB weights nor conversation RAG can answer those, and unanswered they
    # do not come back as "mình không biết" -- they come back as invented
    # detail delivered in the same warm, confident register as everything else.
    # Turning this off restores the previous behaviour exactly: live info and
    # shopping questions still search, knowledge questions go ungrounded.
    web_search_knowledge_enabled: bool = Field(default=True)

    # ── Vision captioning ────────────────────────────────────────────────────
    # `POST /v1/vision/caption` turns an uploaded image into a short text
    # description through the `vision/` pipeline (brain/caption_service.py ->
    # YOLO11n + RapidOCR + CLIP, the graphs already exported to
    # assets/models/vision/). The chat model itself never sees pixels -- this
    # is how an attached image reaches it at all, by becoming words the same
    # Qwen text model can read. Turning this off skips loading those graphs at
    # startup on a machine that never calls the endpoint; the endpoint then
    # answers `caption=""`.
    vision_captioning_enabled: bool = Field(default=True)
    # The ingest is three CPU models in a thread pool with no cancellation
    # point of its own, so this is the only thing standing between a slow
    # caption and a request that never returns. A timeout degrades to
    # `caption=""` -- the same "best-effort enrichment, never a hard failure"
    # contract as RAG/web search -- rather than surfacing as an error the
    # frontend has to handle. Dense screenshots are the slow case: OCR is
    # where the seconds go.
    vision_caption_timeout_seconds: float = Field(default=30.0)

    # ── Knowledge-turn decoding ─────────────────────────────────────────────
    # A grounded factual turn is decoded differently from a chat turn. The
    # conversational 0.65 (with the Modelfile's top_p 0.9) is what makes the
    # same question sample different details on different turns -- fine when
    # the content is rapport, wrong when it is a claim about the world. This is
    # applied by brain.response_policy and composes with the VAD policy by
    # taking whichever asks for the colder, shorter answer.
    knowledge_temperature: float = Field(default=0.30)
    # Deliberately above `llm_max_tokens`: the 320 ceiling is a spoken-latency
    # budget for chat, and a knowledge turn is the one case where the answer is
    # the point. Paired with the 3-5 sentence override in nodes/generate.py --
    # raising this alone would not lengthen anything, since the binding limit
    # is the fine-tune's own "1-3 câu" rule, not num_predict.
    knowledge_max_tokens: int = Field(default=480)

    # ── Memory ───────────────────────────────────────────────────────────────
    # SQLite rows are individual messages, so 6 rows represent approximately
    # three user/assistant exchanges. The store is scoped by `session_id`
    # (rows written before sessions existed read as `default`), so this is
    # purely a prefill/latency budget rather than a leak guard.
    memory_recent_limit: int = Field(default=6)
    # Ceiling on the verbatim history window, in characters. The message count
    # above bounds how many turns are considered; this bounds how much prompt
    # they are allowed to occupy, which is what prefill latency actually tracks.
    memory_recent_char_budget: int = Field(default=1600)
    # Persistent sparse sentence index used to suppress assistant phrasing that
    # closely repeats prior replies. Scores are normalized BM25 similarities.
    bm25_repetition_threshold: float = Field(default=0.78)
    bm25_repetition_min_tokens: int = Field(default=5)
    bm25_repetition_max_sentences: int = Field(default=2000)

    # ── Cloud backend (Gemini via Google AI Studio) ──────────────────────────
    # Read from the GEMINI_API_KEY environment variable / .env. Empty means the
    # graph never routes to cloud at all — `route_backend` treats a missing key
    # as one of its three fallback conditions rather than as an error.
    gemini_api_key: str = Field(default="")
    # Keep this as the one default model name. A user can override it with
    # GEMINI_MODEL when their Google AI Studio account exposes a different one.
    gemini_model: str = Field(default="gemini-3.5-flash-lite")
    # Set false to pin every turn to the local fine-tune without unsetting the
    # key — useful for A/B-ing the two backends on the same conversation.
    cloud_enabled: bool = Field(default=True)
    # Shorter than the local timeout on purpose: a person is waiting to hear a
    # sentence, and falling back to a model that is already resident beats
    # waiting out a slow network.
    cloud_timeout_seconds: float = Field(default=30.0)
    cloud_max_retries: int = Field(default=1)
    # How long a connection failure (not a quota failure) parks the cloud route.
    cloud_transient_cooldown_seconds: float = Field(default=60.0)
    # Free-tier daily quotas roll over at midnight Pacific, so a quota failure
    # parks the route until then rather than for a fixed 24 hours.
    cloud_quota_reset_timezone: str = Field(default="America/Los_Angeles")

    # ── Conversation graph ───────────────────────────────────────────────────
    # Exchanges kept verbatim before the oldest are folded into the session's
    # rolling summary. Distinct from `memory_recent_limit`, which is how many
    # rows the *prompt* carries — that one is a latency budget and stays smaller.
    graph_history_turns: int = Field(default=12)
    graph_summary_max_tokens: int = Field(default=160)
    graph_summary_defer_timeout: float = Field(default=90.0)
    # One retry, matching the existing repetition-retry path. A second would
    # double worst-case latency for a case the logs put at a few percent.
    graph_max_regenerations: int = Field(default=1)

    # ── Reply guard ──────────────────────────────────────────────────────────
    # The persona asks for 1–3 sentences; these are the points past which a
    # reply is judged to have run away from that rather than merely gone long.
    guard_max_sentences: int = Field(default=5)
    guard_max_chars: int = Field(default=600)
    # Ratio against the immediately previous assistant turn, above which the
    # reply counts as the model repeating itself and is regenerated once.
    guard_previous_turn_similarity: float = Field(default=0.85)

    # ── Turn metrics (JSONL) ─────────────────────────────────────────────────
    # One line per turn: route, tokens, latency, guard flags. Separate from
    # mitsuka.log because the questions it answers are counting questions.
    turn_metrics_enabled: bool = Field(default=True)
    turn_metrics_path: Path = Field(
        default=Path(__file__).parent.parent / "logs" / "turns.jsonl"
    )

    # ── Logging ──────────────────────────────────────────────────────────────
    log_level: str = Field(default="INFO")
    # Single-line JSON records (log aggregation) instead of human-readable console.
    log_json: bool = Field(default=False)
    # Optional path for a rotating log file; unset (default) logs to console only.
    log_file: str | None = Field(default=None)
    environment: str = Field(default="development")

    # ── VAD (Valence-Arousal-Dominance) checkpoints ───────────────────────────
    # Trained checkpoints ship in the repo (see model/text_vad.py,
    # model/multimodal_vad.py for the architectures they load into).
    text_vad_checkpoint_path: Path = Field(default=Path("model/best_text_vad.pt"))
    multimodal_vad_checkpoint_path: Path = Field(default=Path("model/best_multimodal_vad.pt"))

    # ── Sherpa-ONNX (speech-to-text, Vietnamese-only) ─────────────────────────
    # Not shipped in the repo (ASR model files are large binaries) — download
    # csukuangfj2/sherpa-onnx-zipformer-vi-30M-int8-2026-02-09 from Hugging Face
    # into `sherpa_onnx_model_dir`, or point the four *_path fields at wherever
    # you already keep it. See README for the exact download command.
    sherpa_onnx_model_dir: Path = Field(
        default=Path(__file__).resolve().parents[3] / "assets" / "models" / "sherpa-onnx-zipformer-vi-30M-int8-2026-02-09"
    )
    sherpa_onnx_tokens: str = Field(default="")
    sherpa_onnx_encoder: str = Field(default="")
    sherpa_onnx_decoder: str = Field(default="")
    sherpa_onnx_joiner: str = Field(default="")
    sherpa_onnx_num_threads: int = Field(default=4)

    # ── Piper (text-to-speech) ─────────────────────────────────────────────────
    # parents[3] from this file (brain/config.py) is
    # apps/local-api/brain -> apps/local-api -> apps -> <repo root>.
    piper_models_dir: Path = Field(default=Path(__file__).resolve().parents[3] / "assets" / "models" / "voices")

    # Voice used when a request asks for "default". Without this the app's voice
    # was whichever one the registry happened to list first.
    tts_default_voice: str = Field(default="fusion_E_ling75_acoustic100")

    # How far the agent's V/A/D is allowed to move tempo, pause length and
    # volume (service/prosody.py). 0.0 renders exactly what Piper would have
    # rendered on its own; 1.0 is the tuned default. Pitch is deliberately not
    # on this dial -- see `tts_pitch_scale`.
    tts_prosody_depth: float = Field(default=1.0, ge=0.0, le=2.0)

    # The voice's pitch, as a multiple of what Piper renders. Constant across
    # every utterance on purpose: it used to be derived per utterance from
    # emotion, and once a reply was spoken sentence by sentence each sentence
    # was scored separately, so one answer came out as a low voice and a high
    # voice alternating. Applied by rendering longer and resampling, which
    # invents nothing; slightly above 1.0 reads as bright rather than thin.
    tts_pitch_scale: float = Field(default=1.06, ge=0.82, le=1.22)

    # Depth of the Fujisaki F0 contour drawn on top of Piper's own intonation.
    # Off by default: it is per-utterance pitch movement, and with per-sentence
    # synthesis every sentence restarted its own declination -- high at the
    # start, low at the end, over and over. Raise it only if replies go back to
    # being synthesised whole.
    tts_contour_depth: float = Field(default=0.0, ge=0.0, le=2.0)

    # ── HTTP server ──────────────────────────────────────────────────────────
    # 8010, not the conventional 8000: Docker Desktop publishes container ports
    # on both `0.0.0.0` and `[::]`, and a stack holding 8000 there takes over
    # `localhost:8000` for everything else on the machine — Windows resolves
    # `localhost` to `::1` first, and uvicorn's `0.0.0.0` is IPv4-only, so the
    # frontend's probe reached the container and saw the API as offline while
    # it was serving fine on `127.0.0.1:8000`.
    api_host: str = Field(default="0.0.0.0")  # noqa: S104 — loopback + LAN, local dev service
    api_port: int = Field(default=8010, ge=1, le=65535)

    # ── CORS ─────────────────────────────────────────────────────────────────
    cors_allow_origins: list[str] = Field(default=["*"])

    # ── Emotion-VAD executor ────────────────────────────────────────────────
    emotion_executor_max_workers: int = Field(default=4)

    @property
    def resolved_text_vad_checkpoint_path(self) -> Path:
        path = self.text_vad_checkpoint_path
        return path if path.is_absolute() else self.base_dir / path

    @property
    def resolved_multimodal_vad_checkpoint_path(self) -> Path:
        path = self.multimodal_vad_checkpoint_path
        return path if path.is_absolute() else self.base_dir / path

    @property
    def resolved_sherpa_tokens(self) -> str:
        return self.sherpa_onnx_tokens or str(self.sherpa_onnx_model_dir / "tokens.txt")

    @property
    def resolved_sherpa_encoder(self) -> str:
        return self.sherpa_onnx_encoder or str(self.sherpa_onnx_model_dir / "encoder.int8.onnx")

    @property
    def resolved_sherpa_decoder(self) -> str:
        return self.sherpa_onnx_decoder or str(self.sherpa_onnx_model_dir / "decoder.onnx")

    @property
    def resolved_sherpa_joiner(self) -> str:
        return self.sherpa_onnx_joiner or str(self.sherpa_onnx_model_dir / "joiner.int8.onnx")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
