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
    ollama_model: str = Field(default="qwen2.5:3b")
    llm_temperature: float = Field(default=0.7)
    llm_max_tokens: int = Field(default=1024)

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

    # ── Memory ───────────────────────────────────────────────────────────────
    memory_recent_limit: int = Field(default=10)

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
    whisper_model: str = Field(default="medium.en")
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
