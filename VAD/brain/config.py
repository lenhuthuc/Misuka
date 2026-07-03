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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
