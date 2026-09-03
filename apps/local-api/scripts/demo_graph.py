"""Interactive smoke test for the production conversation graph."""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

API_ROOT = Path(__file__).resolve().parents[1]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from brain.config import Settings  # noqa: E402
from brain.background import run_memory_tasks  # noqa: E402
from brain.embeddings import EmbeddingModel  # noqa: E402
from brain.llm_service import LLMService  # noqa: E402
from brain.memory_service import MemoryService  # noqa: E402
from brain.rag_service import RAGService  # noqa: E402
from brain.vector_service import VectorService  # noqa: E402
from brain.web_search_service import WebSearchService  # noqa: E402
from core.container import build_conversation_graph  # noqa: E402
from core.llm_priority import LLMPriorityGate  # noqa: E402
from core.tasks import BackgroundTaskRegistry  # noqa: E402


async def run(session_id: str) -> None:
    settings = Settings()
    llm = LLMService(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        reasoning_model=settings.reasoning_model,
    )
    memory = MemoryService(
        settings.sqlite_path,
        bm25_repetition_threshold=settings.bm25_repetition_threshold,
        bm25_repetition_min_tokens=settings.bm25_repetition_min_tokens,
        bm25_repetition_max_sentences=settings.bm25_repetition_max_sentences,
    )
    vector = VectorService(
        embedding_model=EmbeddingModel(
            settings.embedding_model_name, settings.embedding_batch_size
        ),
        collection_name=settings.qdrant_collection,
        qdrant_url=settings.qdrant_url,
        top_k=settings.qdrant_top_k,
    )
    await memory.initialize()
    await vector.initialize()
    rag = RAGService(
        vector=vector,
        rrf_k=settings.rag_rrf_k,
        top_k=settings.qdrant_top_k,
        context_char_budget=settings.rag_context_char_budget,
        min_score=settings.rag_min_score,
    )
    web_search = WebSearchService(
        max_results=settings.web_search_max_results,
        context_char_budget=settings.web_search_context_char_budget,
        timeout_seconds=settings.web_search_timeout_seconds,
        region=settings.web_search_region,
    )
    gate = LLMPriorityGate(speech_lull_seconds=settings.llm_speech_lull_seconds)
    tasks = BackgroundTaskRegistry()
    graph = build_conversation_graph(
        settings,
        llm=llm,
        memory=memory,
        rag=rag,
        web_search=web_search,
        gate=gate,
        tasks=tasks,
    )

    print(f"Mitsuka graph demo (session={session_id!r}); /quit to exit")
    try:
        while True:
            text = (await asyncio.to_thread(input, "You> ")).strip()
            if text.lower() in {"/quit", "/exit"}:
                break
            if not text:
                continue
            async with gate.foreground():
                state = await graph.ainvoke(text, session_id=session_id)
            print(f"Mitsuka> {state['final_reply']}")
            print(
                f"  route={state.get('route', '')} "
                f"flags={state.get('flags', [])} "
                f"guard={state.get('guard_action', 'pass')}"
            )
            # The HTTP endpoints schedule this after returning their response.
            # The CLI has no response lifecycle, so persist synchronously to
            # make the next typed turn see this one. Vector indexing is omitted
            # here to keep an interactive smoke test responsive.
            await run_memory_tasks(
                text,
                state["final_reply"],
                memory,
                session_id=session_id,
            )
    finally:
        await tasks.drain()
        await llm.aclose()
        await memory.close()
        await vector.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-id", default="demo")
    args = parser.parse_args()
    asyncio.run(run(args.session_id))


if __name__ == "__main__":
    main()
