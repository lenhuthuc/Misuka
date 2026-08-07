"""Seed data for the /v1/chat/seed endpoint.

Service construction/teardown now lives in `core.container.ServiceContainer`
— this module used to also host `create_services`/`shutdown_services`, but
having two places that could build the same services was exactly the
duplication REFACTOR_PLAN.md Phase 3 targets.
"""
from typing import Any

SEED_DOCS: list[dict[str, Any]] = [
    {"text": "BrainMaster is an AI chatbot service built on LangGraph with multi-query RAG.",
     "meta": {"source": "readme", "topic": "overview"}},
    {"text": "The vector store uses Qdrant with cosine distance for semantic search.",
     "meta": {"source": "readme", "topic": "vector-store"}},
    {"text": "Conversation history is persisted in an SQLite database via aiosqlite.",
     "meta": {"source": "readme", "topic": "memory"}},
    {"text": "Embeddings are generated on CPU using sentence-transformers with optional ONNX acceleration.",
     "meta": {"source": "readme", "topic": "embeddings"}},
    {"text": "The LLM is served locally via Ollama running qwen2.5:3b.",
     "meta": {"source": "readme", "topic": "llm"}},
]
