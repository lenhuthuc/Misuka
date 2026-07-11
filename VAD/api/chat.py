from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from brain.app import SEED_DOCS
from brain.background import run_memory_tasks
from brain.emotion_service import EmotionReading, EmotionService, extract_memory_vads
from brain.nodes.generate import build_messages
from brain.nodes.should_rag import should_use_rag
from schemas.vad import VADScores

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/chat", tags=["chat"])


class ChatRequest(BaseModel):
    query: str


class ChatResponse(BaseModel):
    response: str
    generated_queries: list[str]
    docs_count: int
    # Current system emotional state: response V/A/D blended with retrieved memories' V/A/D
    emotion: str
    state: VADScores


class SeedResponse(BaseModel):
    inserted: int


async def _emotion_state(
    emotion_svc: EmotionService,
    response: str,
    docs: list,
) -> tuple[EmotionReading, EmotionReading]:
    """Infer the response's own V/A/D, then blend with retrieved memories' V/A/D.

    Returns (response_reading, blended_current_state).
    """
    reading = await emotion_svc.infer(response)
    state = emotion_svc.current_state(reading, extract_memory_vads(docs))
    return reading, state


@router.post("", response_model=ChatResponse)
async def chat(body: ChatRequest, request: Request) -> ChatResponse:
    graph = request.app.state.brain_graph
    result = await graph.ainvoke(
        {
            "messages": [],
            "query": body.query,
            "generated_queries": [],
            "retrieved_docs": [],
            "context": "",
            "response": "",
            "use_rag": False,       # will be set by should_rag_node
            "should_save_facts": False,
        }
    )

    response_text = result["response"]
    reading, state = await _emotion_state(
        request.app.state.brain_emotion, response_text, result.get("retrieved_docs", [])
    )

    # Save conversation history and optionally extract long-term facts in the
    # background — client receives the response without waiting for these.
    llm = request.app.state.brain_llm
    memory = request.app.state.brain_memory
    vector = request.app.state.brain_vector

    def _on_bg_done(task: asyncio.Task) -> None:
        if exc := task.exception():
            logger.warning("background memory task failed: %s", exc)

    bg = asyncio.create_task(
        run_memory_tasks(body.query, response_text, memory, llm, emotion=reading, vector=vector)
    )
    bg.add_done_callback(_on_bg_done)

    return ChatResponse(
        response=response_text,
        generated_queries=result.get("generated_queries", []),
        docs_count=len(result.get("retrieved_docs", [])),
        emotion=state.emotion,
        state=VADScores(valence=state.valence, arousal=state.arousal, dominance=state.dominance),
    )


@router.post("/stream")
async def chat_stream(body: ChatRequest, request: Request) -> StreamingResponse:
    """SSE endpoint: streams token-by-token from Ollama, fires memory tasks in background.

    All pre-processing (RAG, message building) runs inside the generator so that
    response headers are sent immediately — the browser never hangs waiting for them.
    """
    llm = request.app.state.brain_llm
    memory = request.app.state.brain_memory
    rag = request.app.state.brain_rag

    async def event_generator():
        # Establish SSE connection immediately — browser receives headers right away
        yield ": ping\n\n"

        full_response = ""
        docs = []
        try:
            context = ""
            if should_use_rag(body.query):
                try:
                    _, docs, context = await rag.build_context(body.query)
                except Exception as exc:
                    logger.warning("chat_stream | RAG failed, skipping: %s", exc)

            messages = await build_messages(body.query, context, memory)

            async for chunk in llm.stream_chat(messages):
                full_response += chunk
                yield f"data: {json.dumps({'content': chunk})}\n\n"

        except Exception as exc:
            logger.warning("chat_stream | error: %s", exc)
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

        reading = None
        if full_response:
            try:
                reading, state = await _emotion_state(
                    request.app.state.brain_emotion, full_response, docs
                )
                state_payload = {
                    "emotion": state.emotion,
                    "state": VADScores(
                        valence=state.valence, arousal=state.arousal, dominance=state.dominance
                    ).model_dump(),
                }
                yield f"data: {json.dumps(state_payload)}\n\n"
            except Exception as exc:
                logger.warning("chat_stream | emotion inference failed: %s", exc)

        yield "data: [DONE]\n\n"

        if full_response:
            def _on_bg_done(task: asyncio.Task) -> None:
                if exc := task.exception():
                    logger.warning("background memory task failed: %s", exc)

            bg = asyncio.create_task(
                run_memory_tasks(
                    body.query, full_response, memory, llm,
                    emotion=reading, vector=request.app.state.brain_vector,
                )
            )
            bg.add_done_callback(_on_bg_done)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.post("/seed", response_model=SeedResponse)
async def seed(request: Request) -> SeedResponse:
    """Seed Qdrant with built-in sample documents."""
    vector = request.app.state.brain_vector
    texts = [d["text"] for d in SEED_DOCS]
    metas = [d["meta"] for d in SEED_DOCS]
    ids = await vector.upsert(texts, metas)
    return SeedResponse(inserted=len(ids))
