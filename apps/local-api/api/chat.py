from __future__ import annotations

import asyncio
import logging
import uuid

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from api.dependencies import get_container
from brain.app import SEED_DOCS
from brain.background import run_memory_tasks
from brain.emotion_service import EmotionReading, EmotionService, extract_memory_vads
from brain.response_policy import ResponsePolicy
from core.container import ServiceContainer
from core.logging import bind_turn_id
from schemas.chat import (
    ChatStreamDeltaEvent,
    ChatStreamDoneEvent,
    ChatStreamEmotionEvent,
    ChatStreamErrorDetail,
    ChatStreamErrorEvent,
)
from schemas.vad import AgentVAD, VADScores

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/chat", tags=["chat"])


class ChatRequest(BaseModel):
    query: str
    session_id: str = "default"
    # Produced by the client-side audio/text VAD fusion when available.
    # Omitting it preserves the existing chat behaviour exactly.
    user_vad: VADScores | None = None


class ChatResponse(BaseModel):
    turn_id: str
    response: str
    generated_queries: list[str]
    docs_count: int
    # Current system emotional state: response V/A/D blended with retrieved memories' V/A/D
    emotion: str
    state: VADScores
    # Text-only V/A/D for `response`, in [0, 1] — see schemas/vad.py:AgentVAD.
    agent_vad: AgentVAD
    response_policy: "ResponsePolicyResponse | None" = None


class ResponsePolicyResponse(BaseModel):
    instruction: str
    max_tokens: int | None
    temperature: float | None
    stream_pace: str

    @classmethod
    def from_policy(cls, policy: ResponsePolicy) -> "ResponsePolicyResponse":
        return cls(
            instruction=policy.instruction,
            max_tokens=policy.max_tokens,
            temperature=policy.temperature,
            stream_pace=policy.stream_pace,
        )


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


async def _agent_vad(container: ServiceContainer, response: str) -> AgentVAD:
    """Text-only V/A/D for the agent's complete response — never touches user
    audio, the user transcript, user_vad, or WavLM embeddings."""
    v, a, d = await asyncio.to_thread(container.emotion_pipeline.analyze_agent_response, response)
    return AgentVAD(valence=v, arousal=a, dominance=d)


@router.post("", response_model=ChatResponse)
async def chat(body: ChatRequest, container: ServiceContainer = Depends(get_container)) -> ChatResponse:
    turn_id = str(uuid.uuid4())
    with bind_turn_id(turn_id):
        async with container.llm_gate.foreground():
            graph_state = await container.graph.ainvoke(
                body.query,
                session_id=body.session_id,
                turn_id=turn_id,
                user_vad=body.user_vad,
            )

        response_text = graph_state["final_reply"]
        docs = graph_state.get("retrieved_docs", [])
        policy = graph_state["response_policy"]
        reading, state = await _emotion_state(container.emotion, response_text, docs)
        agent_vad = await _agent_vad(container, response_text)

        # Save and index the exchange in the background — the client receives
        # the response without waiting for either. Spawned while turn_id is still bound, so the background
        # task's own log lines (see core/tasks.py) inherit it too. The gate is
        # what keeps the exchange's embedding from running while the reply is
        # still being synthesised and spoken.
        container.tasks.spawn(
            run_memory_tasks(
                body.query, response_text, container.memory,
                emotion=reading, vector=container.vector,
                gate=container.llm_gate, defer_timeout=container.index_defer_timeout,
                session_id=body.session_id,
            ),
            name="chat.memory_tasks",
        )

    return ChatResponse(
        turn_id=turn_id,
        response=response_text,
        generated_queries=graph_state.get("generated_queries", []),
        docs_count=len(docs),
        emotion=state.emotion,
        state=VADScores(valence=state.valence, arousal=state.arousal, dominance=state.dominance),
        agent_vad=agent_vad,
        response_policy=ResponsePolicyResponse.from_policy(policy) if policy.is_active else None,
    )


@router.post("/stream")
async def chat_stream(body: ChatRequest, container: ServiceContainer = Depends(get_container)) -> StreamingResponse:
    """Run the same conversation graph as `/v1/chat`, exposing guarded deltas as SSE."""
    turn_id = str(uuid.uuid4())

    async def event_generator():
        with bind_turn_id(turn_id):
            # Establish SSE connection immediately — browser receives headers right away
            yield ": ping\n\n"

            full_response = ""
            docs: list = []
            graph_state = None
            try:
                # Gate covers retrieval too, not just the stream: RAG takes long
                # enough that a background generation slipping in during it would
                # still land in front of this turn's first token.
                async with container.llm_gate.foreground():
                    async for kind, payload in container.graph.astream(
                        body.query,
                        session_id=body.session_id,
                        turn_id=turn_id,
                        user_vad=body.user_vad,
                    ):
                        if kind == "delta":
                            event = ChatStreamDeltaEvent(turn_id=turn_id, content=str(payload))
                            yield f"data: {event.model_dump_json()}\n\n"
                        elif kind == "state":
                            graph_state = payload

                    if graph_state is not None:
                        full_response = graph_state.get("final_reply", "")
                        docs = graph_state.get("retrieved_docs", [])
                    elif not full_response:
                        logger.error("chat_stream | graph completed without final state")

            except Exception as exc:
                logger.exception("chat_stream | LLM stream failed")
                error_event = ChatStreamErrorEvent(
                    turn_id=turn_id,
                    error=ChatStreamErrorDetail(code="LLM_UNAVAILABLE", message=str(exc), retryable=True),
                )
                yield f"data: {error_event.model_dump_json()}\n\n"

            reading = None
            agent_vad = None
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

                try:
                    agent_vad = await _agent_vad(container, full_response)
                except Exception:
                    logger.exception("chat_stream | agent_vad inference failed")

            yield f"data: {ChatStreamDoneEvent(turn_id=turn_id, agent_vad=agent_vad).model_dump_json()}\n\n"

            if full_response:
                container.tasks.spawn(
                    run_memory_tasks(
                        body.query, full_response, container.memory,
                        emotion=reading, vector=container.vector,
                        gate=container.llm_gate, defer_timeout=container.index_defer_timeout,
                        session_id=body.session_id,
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
