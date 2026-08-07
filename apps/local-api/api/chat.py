from __future__ import annotations

import logging
import uuid
from collections.abc import Callable

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from api.dependencies import get_container
from application.conversation_turn import prepare_turn
from brain.app import SEED_DOCS
from brain.background import run_memory_tasks
from brain.emotion_service import EmotionReading, EmotionService, extract_memory_vads
from core.container import ServiceContainer
from core.logging import bind_turn_id, log_duration
from schemas.chat import (
    ChatStreamDeltaEvent,
    ChatStreamDoneEvent,
    ChatStreamEmotionEvent,
    ChatStreamErrorDetail,
    ChatStreamErrorEvent,
)
from schemas.vad import VADScores

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/chat", tags=["chat"])


class ChatRequest(BaseModel):
    query: str


class ChatResponse(BaseModel):
    turn_id: str
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


def _log_rag_error(operation: str) -> Callable[[Exception], None]:
    def _handler(exc: Exception) -> None:
        # Non-fatal (the turn degrades to no-context) but still an unexpected
        # backend failure — keep the traceback server-side.
        logger.exception("%s | RAG failed, skipping", operation)
    return _handler


@router.post("", response_model=ChatResponse)
async def chat(body: ChatRequest, container: ServiceContainer = Depends(get_container)) -> ChatResponse:
    turn_id = str(uuid.uuid4())
    with bind_turn_id(turn_id):
        turn = await prepare_turn(
            body.query, container.memory, container.rag, container.memory_recent_limit,
            on_rag_error=_log_rag_error("chat"),
        )
        with log_duration(logger, "llm.chat", component="llm"):
            response_text = await container.llm.chat(turn.messages)

        reading, state = await _emotion_state(container.emotion, response_text, turn.retrieved_docs)

        # Save conversation history and optionally extract long-term facts in
        # the background — client receives the response without waiting for
        # these. Spawned while turn_id is still bound, so the background
        # task's own log lines (see core/tasks.py) inherit it too.
        container.tasks.spawn(
            run_memory_tasks(
                body.query, response_text, container.memory, container.llm,
                emotion=reading, vector=container.vector,
            ),
            name="chat.memory_tasks",
        )

    return ChatResponse(
        turn_id=turn_id,
        response=response_text,
        generated_queries=turn.generated_queries,
        docs_count=len(turn.retrieved_docs),
        emotion=state.emotion,
        state=VADScores(valence=state.valence, arousal=state.arousal, dominance=state.dominance),
    )


@router.post("/stream")
async def chat_stream(body: ChatRequest, container: ServiceContainer = Depends(get_container)) -> StreamingResponse:
    """SSE endpoint: streams token-by-token from Ollama, fires memory tasks in background.

    All pre-processing (RAG, message building) runs inside the generator so that
    response headers are sent immediately — the browser never hangs waiting for them.
    Uses the same `prepare_turn` policy as the buffered `/v1/chat` endpoint.
    """
    turn_id = str(uuid.uuid4())

    async def event_generator():
        with bind_turn_id(turn_id):
            # Establish SSE connection immediately — browser receives headers right away
            yield ": ping\n\n"

            full_response = ""
            docs: list = []
            try:
                turn = await prepare_turn(
                    body.query, container.memory, container.rag, container.memory_recent_limit,
                    on_rag_error=_log_rag_error("chat_stream"),
                )
                docs = turn.retrieved_docs

                stream_start_logged = False
                async for chunk in container.llm.stream_chat(turn.messages):
                    if not stream_start_logged:
                        logger.debug("llm.stream_chat | first token received", extra={"component": "llm"})
                        stream_start_logged = True
                    full_response += chunk
                    event = ChatStreamDeltaEvent(turn_id=turn_id, content=chunk)
                    yield f"data: {event.model_dump_json()}\n\n"

            except Exception as exc:
                logger.exception("chat_stream | LLM stream failed")
                error_event = ChatStreamErrorEvent(
                    turn_id=turn_id,
                    error=ChatStreamErrorDetail(code="LLM_UNAVAILABLE", message=str(exc), retryable=True),
                )
                yield f"data: {error_event.model_dump_json()}\n\n"

            reading = None
            if full_response:
                try:
                    reading, state = await _emotion_state(
                        container.emotion, full_response, docs
                    )
                    emotion_event = ChatStreamEmotionEvent(
                        turn_id=turn_id,
                        emotion=state.emotion,
                        state=VADScores(valence=state.valence, arousal=state.arousal, dominance=state.dominance),
                    )
                    yield f"data: {emotion_event.model_dump_json()}\n\n"
                except Exception:
                    logger.exception("chat_stream | emotion inference failed")

            yield f"data: {ChatStreamDoneEvent(turn_id=turn_id).model_dump_json()}\n\n"

            if full_response:
                container.tasks.spawn(
                    run_memory_tasks(
                        body.query, full_response, container.memory, container.llm,
                        emotion=reading, vector=container.vector,
                    ),
                    name="chat_stream.memory_tasks",
                )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
            "X-Turn-Id": turn_id,
        },
    )


@router.post("/seed", response_model=SeedResponse)
async def seed(container: ServiceContainer = Depends(get_container)) -> SeedResponse:
    """Seed Qdrant with built-in sample documents."""
    texts = [d["text"] for d in SEED_DOCS]
    metas = [d["meta"] for d in SEED_DOCS]
    ids = await container.vector.upsert(texts, metas)
    return SeedResponse(inserted=len(ids))
