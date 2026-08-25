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
    # 13.1 tok/s for 1.5b — an exact 2x for an exact 2x in weights. That makes
    # file size, not parameter count, the thing to shop on: a newer model of
    # the same size is free.
    #
    # Benchmarked against this app's real system prompt, four spoken turns each
    # (story, a playful follow-up inside that story, arithmetic, chit-chat):
    #   qwen2.5:1.5b  degenerates on any turn needing content — announces a
    #                 story, invents a meta-title, stalls.
    #   qwen2.5:3b    answered the story turn in *Chinese*. Rule 1 says
    #                 Vietnamese only; a model that breaks it is not a
    #                 candidate however fluent the rest is.
    #   qwen3:4b      2.5GB, and still reasons aloud in English with thinking
    #                 disabled — spent the whole token ceiling on it in 3 of 4
    #                 turns and never reached a reply.
    #   gemma3:4b     best Vietnamese of the four and the warmest in character,
    #                 but 3.3GB is 2.4x the weights of the pick below.
    #   qwen3:1.7b    clean Vietnamese in all four, held the 1-3 sentence rule,
    #                 and played along with the joke turn.
    # 1.7b wins on being *smaller* than the 3b it replaces (1.4GB vs 1.9GB), so
    # it is the rare change that is faster and better at once. Being a Qwen3 it
    # must have thinking off — see `_THINK` in brain/llm_service.py.
    ollama_model: str = Field(default="qwen3:1.7b")
    llm_temperature: float = Field(default=0.7)
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
    rag_context_char_budget: int = Field(default=2000)
    # Cosine floor a hit must clear to reach the prompt. Qdrant always returns
    # its `top_k` nearest points, however far away they are, so an unrelated
    # question still retrieved five memories and spent ~900 characters of
    # prefill on them. Fusion ranks hits against each other and cannot tell
    # "best of a bad lot" from "relevant", so the floor has to be applied to
    # the raw similarity, before RRF. 0.0 disables it.
    rag_min_score: float = Field(default=0.35)

    # ── Memory ───────────────────────────────────────────────────────────────
    memory_recent_limit: int = Field(default=10)
    # Ceiling on the verbatim history window, in characters. The message count
    # above bounds how many turns are considered; this bounds how much prompt
    # they are allowed to occupy, which is what prefill latency actually tracks.
    memory_recent_char_budget: int = Field(default=3000)

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
