"""
Test RAG pipeline ở 2 mức:
  Level 1 — API  : dùng HTTP (server phải đang chạy tại localhost:8000)
  Level 2 — Unit : khởi tạo service trực tiếp (không cần server, cần Ollama)

Chạy:
    cd D:\myProject\Mitsuka\VAD
    python scripts/test_rag.py              # cả 2 mức
    python scripts/test_rag.py --api-only   # chỉ HTTP
    python scripts/test_rag.py --unit-only  # chỉ unit
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

BASE = "http://localhost:8000"
OLLAMA = "http://localhost:11434"


# ═══════════════════════════════════════════════════════════════════════════════
# LEVEL 1 — API tests (qua HTTP)
# ═══════════════════════════════════════════════════════════════════════════════

def test_api_seed():
    print("\n[API] POST /v1/chat/seed")
    r = requests.post(f"{BASE}/v1/chat/seed")
    assert r.status_code == 200, f"HTTP {r.status_code}: {r.text}"
    body = r.json()
    assert "inserted" in body
    print(f"  [PASS] inserted={body['inserted']} docs into Qdrant")
    return body["inserted"]


def test_api_chat(query: str, expect_keywords: list[str] | None = None):
    print(f"\n[API] POST /v1/chat  query={repr(query)}")
    t0 = time.perf_counter()
    r = requests.post(f"{BASE}/v1/chat", json={"query": query}, timeout=60)
    elapsed = time.perf_counter() - t0

    assert r.status_code == 200, f"HTTP {r.status_code}: {r.text}"
    body = r.json()

    assert "response" in body
    assert "generated_queries" in body
    assert "docs_count" in body

    print(f"  queries expanded : {body['generated_queries']}")
    print(f"  docs retrieved   : {body['docs_count']}")
    print(f"  response ({elapsed:.1f}s) : {body['response'][:200]}")

    if expect_keywords:
        lower = body["response"].lower()
        for kw in expect_keywords:
            found = kw.lower() in lower
            print(f"  keyword {repr(kw):20s}: {'[PASS]' if found else '[WARN] not found'}")

    print(f"  [PASS] /v1/chat returned valid response")
    return body


def run_api_tests():
    print("\n" + "=" * 60)
    print("LEVEL 1 — API tests")
    print("=" * 60)

    # Seed
    n = test_api_seed()
    assert n > 0, "No docs inserted"

    # Query về nội dung đã seed
    test_api_chat(
        "What is BrainMaster and how does it work?",
        expect_keywords=["brainmaster", "langgraph", "rag"],
    )

    # Query về vector store
    test_api_chat(
        "What database is used for storing vectors?",
        expect_keywords=["qdrant"],
    )

    # Query không liên quan — model vẫn phải trả lời trung thực
    test_api_chat("What is the capital of France?")

    print("\n  [PASS] All API tests passed")


# ═══════════════════════════════════════════════════════════════════════════════
# LEVEL 2 — Unit tests (khởi tạo service trực tiếp)
# ═══════════════════════════════════════════════════════════════════════════════

async def test_unit_embedding():
    print("\n[Unit] EmbeddingModel")
    from brain.embeddings import EmbeddingModel
    import numpy as np

    em = EmbeddingModel()

    vecs = em.embed(["hello world", "xin chào thế giới", "this is a test"])
    assert vecs.shape == (3, em.dimension), f"shape mismatch: {vecs.shape}"
    assert vecs.dtype == np.float32

    # Cosine sim: "hello world" vs "xin chào thế giới" phải > 0 (đều nói về greeting)
    cos = float(np.dot(vecs[0], vecs[1]))
    print(f"  dim={em.dimension}  cosine('hello','xin chào')={cos:.3f}")
    print(f"  [PASS] embed() returns ({len(vecs)}, {em.dimension}) float32")

    vec1 = await em.aembed_one("async test")
    assert vec1.shape == (em.dimension,)
    print(f"  [PASS] aembed_one() async works")


async def test_unit_vector_store():
    print("\n[Unit] VectorService (in-memory Qdrant)")
    from brain.embeddings import EmbeddingModel
    from brain.vector_service import VectorService

    em = EmbeddingModel()
    vs = VectorService(embedding_model=em, collection_name="test_col", top_k=3)
    await vs.initialize()

    docs = [
        "FastAPI is a modern Python web framework",
        "Qdrant is a vector similarity search engine",
        "LangGraph builds stateful multi-actor AI applications",
        "SQLite is a lightweight embedded database",
        "Sentence Transformers compute dense vector representations",
    ]
    ids = await vs.upsert(docs, metadatas=[{"idx": i} for i in range(len(docs))])
    assert len(ids) == len(docs)
    print(f"  upserted {len(ids)} docs")

    # Search
    results = await vs.search("what is a vector database?", top_k=2)
    assert len(results) > 0
    top = results[0].payload.get("content", "")
    print(f"  top result: {repr(top[:80])}")
    print(f"  [PASS] upsert + search OK")

    # Batch search
    batch = await vs.search_batch(["web framework", "database"], top_k=2)
    assert len(batch) == 2
    print(f"  [PASS] search_batch({len(batch)} queries) OK")

    await vs.close()


async def test_unit_rag():
    print("\n[Unit] RAGService (needs Ollama)")
    from brain.embeddings import EmbeddingModel
    from brain.llm_service import LLMService
    from brain.vector_service import VectorService
    from brain.rag_service import RAGService

    em  = EmbeddingModel()
    llm = LLMService(base_url=OLLAMA, model="qwen2.5:3b")
    vs  = VectorService(embedding_model=em, collection_name="rag_test", top_k=3)
    await vs.initialize()

    # Seed a few docs
    seed_docs = [
        "BrainMaster uses LangGraph for orchestration",
        "RAG stands for Retrieval Augmented Generation",
        "Multi-query expansion improves recall by generating query variants",
        "Reciprocal Rank Fusion (RRF) merges ranked lists from multiple queries",
    ]
    await vs.upsert(seed_docs)
    print(f"  seeded {len(seed_docs)} docs")

    rag = RAGService(llm=llm, vector=vs, num_queries=2, top_k=3)

    # Test retrieve only
    print("  retrieving...")
    t0 = time.perf_counter()
    queries, docs, context = await rag.build_context("How does multi-query RAG work?")
    elapsed = time.perf_counter() - t0

    print(f"  generated queries : {queries}")
    print(f"  retrieved docs    : {len(docs)}")
    print(f"  context snippet   : {context[:150].replace(chr(10), ' ')}")
    print(f"  elapsed           : {elapsed:.1f}s")
    assert len(docs) > 0, "No docs retrieved"
    assert len(queries) >= 1
    print(f"  [PASS] RAGService.build_context() OK")

    await vs.close()
    await llm.aclose()


async def run_unit_tests():
    print("\n" + "=" * 60)
    print("LEVEL 2 — Unit tests")
    print("=" * 60)

    await test_unit_embedding()
    await test_unit_vector_store()
    await test_unit_rag()

    print("\n  [PASS] All unit tests passed")


# ═══════════════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════════════

async def main(args):
    if not args.unit_only:
        try:
            run_api_tests()
        except requests.exceptions.ConnectionError:
            print("  [SKIP] Server not reachable at localhost:8000 — start with: uvicorn main:app")

    if not args.api_only:
        await run_unit_tests()

    print("\n=== RAG tests done ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-only",  action="store_true")
    parser.add_argument("--unit-only", action="store_true")
    args = parser.parse_args()
    asyncio.run(main(args))
