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
    # 13.1 tok/s for 1.5b — an exact 2x for an exact 2x in weights. 1.5b is the
    # accuracy/latency compromise that keeps spoken turns responsive.
    ollama_model: str = Field(default="qwen2.5:1.5b")
    llm_temperature: float = Field(default=0.7)
    # A spoken turn that runs past a few sentences costs twice: once to decode
    # now (~13 tok/s on this CPU) and again on every later turn, since the reply
    # is re-sent inside the history window. 1024 allowed 2,900-character
    # answers that pushed prompts past 11k characters after five turns.
    llm_max_tokens: int = Field(default=320)

    # ── Memory curator (background fact extraction) ──────────────────────────
    # Not the smallest model available, deliberately. Measured on fact-rich
    # input, qwen2.5:0.5b answered "None" even when the prompt asked for facts
    # alone, while 1.5b extracted seven usable ones. Reusing the chat model
    # keeps this free of extra RAM; the cost is that a curator call evicts the
    # chat model's cached prompt prefix, worth roughly 1s on the following turn.
    curator_model: str = Field(default="qwen2.5:1.5b")
    # Exchanges mined per LLM call. Batching amortises prefill, which is the
    # dominant cost; too large a batch is a longer uninterruptible window.
    curator_batch_size: int = Field(default=5)
    # How long a batch waits for a quiet runner before giving up this round.
    curator_idle_timeout: float = Field(default=30.0)
    # Back-off after a round found no quiet moment. The queue is durable, so
    # waiting costs recall lag rather than lost memories.
    curator_retry_seconds: float = Field(default=20.0)
    # A batch the curator keeps failing on is poison; stop retrying it.
    curator_max_attempts: int = Field(default=3)

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

    # ── Memory ───────────────────────────────────────────────────────────────
    memory_recent_limit: int = Field(default=10)
    # Ceiling on the verbatim history window, in characters. The message count
    # above bounds how many turns are considered; this bounds how much prompt
    # they are allowed to occupy, which is what prefill latency actually tracks.
    memory_recent_char_budget: int = Field(default=3000)
    # Ceiling on the facts block in the system prompt. Facts are the only
    # memory that survives across sessions, so they earn prompt space — but the
    # table only grows and every turn re-sends all of it.
    memory_facts_char_budget: int = Field(default=600)

    # ── Logging ──────────────────────────────────────────────────────────────
    log_level: str = Field(default="INFO")
    # Single-line JSON records (log aggregation) instead of human-readable console.
    log_json: bool = Field(default=False)
    # Optional path for a rotating log file; unset (default) logs to console only.
    log_file: str | None = Field(default=None)
    environment: str = Field(default="development")

    # ── VAD (Valence-Arousal-Dominance) text model ────────────────────────────
    vad_model_path: Path = Field(default=Path("model/vad_bert_final.pt"))

    # ── Whisper (speech-to-text) ───────────────────────────────────────────────
    # `small.en` is the accuracy/latency compromise for accented English on
    # CPU. `base.en` was fast but too error-prone for speaking practice.
    whisper_model: str = Field(default="small.en")
    whisper_device: str = Field(default="cpu")
    whisper_compute: str = Field(default="int8")
    whisper_models_dir: Path = Field(default=Path(__file__).parent.parent / "models")

    # ── Piper (text-to-speech) ─────────────────────────────────────────────────
    # parents[3] from this file (brain/config.py) is
    # apps/local-api/brain -> apps/local-api -> apps -> <repo root>.
    piper_models_dir: Path = Field(default=Path(__file__).resolve().parents[3] / "assets" / "models" / "voices")

    # ── CORS ─────────────────────────────────────────────────────────────────
    cors_allow_origins: list[str] = Field(default=["*"])

    # ── Emotion-VAD executor ────────────────────────────────────────────────
    emotion_executor_max_workers: int = Field(default=4)

    @property
    def resolved_vad_model_path(self) -> Path:
        return self.vad_model_path if self.vad_model_path.is_absolute() else self.base_dir / self.vad_model_path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
