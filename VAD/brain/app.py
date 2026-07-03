"""Brain service factory — called once from the FastAPI lifespan."""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

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


async def create_services():
    """Initialize all Brain services. Returns (llm, memory, vector, rag, fast_graph)."""
    from brain.config import get_settings
    from brain.graph import build_fast_graph
    from brain.llm_service import LLMService
    from brain.memory_service import MemoryService
    from brain.vector_service import VectorService
    from brain.rag_service import RAGService
    from brain.embeddings import EmbeddingModel

    cfg = get_settings()
    embedder = EmbeddingModel(cfg.embedding_model_name, cfg.embedding_batch_size)

    llm = LLMService(
        base_url=cfg.ollama_base_url,
        model=cfg.ollama_model,
        temperature=cfg.llm_temperature,
        max_tokens=cfg.llm_max_tokens,
    )

    memory = MemoryService(cfg.sqlite_path)
    await memory.initialize()

    vector = VectorService(
        embedding_model=embedder,
        collection_name=cfg.qdrant_collection,
        qdrant_url=cfg.qdrant_url,
        top_k=cfg.qdrant_top_k,
    )
    await vector.initialize()

    rag = RAGService(
        vector=vector,
        rrf_k=cfg.rag_rrf_k,
        top_k=cfg.qdrant_top_k,
    )

    # Fast graph: retrieve + generate only — decide/memory run in background
    fast_graph = build_fast_graph(llm, memory, rag, recent_limit=cfg.memory_recent_limit)
    logger.info("Brain services initialized")
    return llm, memory, vector, rag, fast_graph


async def shutdown_services(llm, memory, vector) -> None:
    """Teardown — call inside FastAPI lifespan shutdown block."""
    await llm.aclose()
    await memory.close()
    await vector.close()
    logger.info("Brain services shut down")
