from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Callable

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from api.dependencies import get_container
from application.conversation_turn import prepare_turn
from brain.bm25_repetition import take_complete_sentences
from brain.trailing_question import TrailingQuestionSuppressor, strip_trailing_question
from brain.reasoning_policy import inject_reasoning_note
from brain.app import SEED_DOCS
from brain.background import run_memory_tasks
from brain.emotion_service import EmotionReading, EmotionService, extract_memory_vads
from brain.response_policy import ResponsePolicy, derive_response_policy
from core.container import ServiceContainer
from core.logging import bind_turn_id, log_duration
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


def _log_rag_error(operation: str) -> Callable[[Exception], None]:
    def _handler(exc: Exception) -> None:
        # Non-fatal (the turn degrades to no-context) but still an unexpected
        # backend failure — keep the traceback server-side.
        logger.exception("%s | RAG failed, skipping", operation)
    return _handler


def _log_web_search_error(operation: str) -> Callable[[Exception], None]:
    def _handler(exc: Exception) -> None:
        # Same contract as _log_rag_error: the turn degrades to no web
        # context rather than failing outright.
        logger.exception("%s | web search failed, skipping", operation)
    return _handler


def _filter_repeated_response(container: ServiceContainer, text: str, *, fallback: str = "") -> str:
    filter_response = getattr(container.memory, "filter_assistant_response", None)
    if filter_response is None:
        return text
    return filter_response(text, fallback=fallback)


# Sent only on the retry, after BM25 has already judged a whole reply to be
# something the assistant has said before. It states the finding rather than
# adding another standing rule to the turn note: the standing rules are read on
# every turn and this one is true on roughly seven in a hundred, so carrying it
# always would spend prefill on every turn to describe a rare event -- and give
# the model one more prohibition to satisfy with a question.
_REPETITION_RETRY_NOTE = (
    "Câu trả lời bạn vừa viết lặp lại gần như nguyên văn một câu bạn đã nói "
    "trước đó trong cuộc trò chuyện này, nên nó bị bỏ. Viết lại bằng nội dung "
    "khác hẳn: trả lời thẳng vào điều người dùng vừa nói, không hỏi lại, không "
    "lặp lại lời mời hay câu hỏi cũ."
)


def _with_repetition_note(messages: list[dict[str, str]]) -> list[dict[str, str]] | None:
    """Same messages, plus the retry note in the position that works.

    Returns None when there is nowhere safe to put it. The note has to land
    immediately before the user's question -- see the layout rule at the top of
    `brain/nodes/generate.py`: after the user message the model reads the note
    as the user's words, and at index 0 it silently replaces the Modelfile
    persona. On a turn with no history there is no such position, so that turn
    keeps the old behaviour rather than losing the persona to a retry.
    """
    if len(messages) < 2:
        return None
    head, user = list(messages[:-1]), messages[-1]
    if head[-1]["role"] == "system":
        # Merge, so the note stays the last thing read before the question.
        head[-1] = {**head[-1], "content": f"{head[-1]['content']}\n\n{_REPETITION_RETRY_NOTE}"}
    else:
        head.append({"role": "system", "content": _REPETITION_RETRY_NOTE})
    return [*head, user]


async def _regenerate_without_repetition(
    container: ServiceContainer,
    messages: list[dict[str, str]],
    options: dict,
    operation: str,
) -> str:
    """One more attempt at a reply BM25 rejected in full. "" if it cannot help.

    Only ever reached when the filter emptied the reply completely, which the
    logs put at a few percent of turns -- so the extra generation is paid
    rarely, and only where the alternative is repeating a sentence back at the
    user word for word.
    """
    retry_messages = _with_repetition_note(messages)
    if retry_messages is None:
        return ""
    try:
        retry_text = await container.llm.chat(retry_messages, options=options)
    except Exception:
        logger.exception("%s | repetition retry failed, keeping the original reply", operation)
        return ""
    kept = _filter_repeated_response(container, retry_text)
    if kept.strip():
        logger.info("%s | repetition retry produced a fresh reply", operation)
        return kept
    logger.info("%s | repetition retry repeated too, falling back to raw text", operation)
    return ""


async def _with_optional_reasoning(container: ServiceContainer, turn, operation: str) -> list[dict[str, str]]:
    """Run the hidden deliberation pass without making it a turn dependency.

    A missing/unavailable reasoning model degrades to the normal fast answer;
    it must never turn an otherwise answerable chat request into a 5xx.
    """
    if not turn.reasoning.enabled:
        return turn.messages
    try:
        reasoning = await container.llm.reason(
            turn.messages,
            max_tokens=turn.reasoning.token_budget,
        )
    except Exception:
        logger.exception("%s | adaptive reasoning failed, answering directly", operation)
        return turn.messages
    return inject_reasoning_note(turn.messages, reasoning)


@router.post("", response_model=ChatResponse)
async def chat(body: ChatRequest, container: ServiceContainer = Depends(get_container)) -> ChatResponse:
    turn_id = str(uuid.uuid4())
    with bind_turn_id(turn_id):
        async with container.llm_gate.foreground():
            policy = derive_response_policy(body.user_vad, container.llm.max_tokens)
            turn = await prepare_turn(
                body.query, container.memory, container.rag, container.memory_recent_limit,
                history_char_budget=container.memory_recent_char_budget,
                on_rag_error=_log_rag_error("chat"),
                response_policy=policy,
                reasoning_enabled=container.reasoning_enabled,
                reasoning_activation_threshold=container.reasoning_activation_threshold,
                reasoning_min_tokens=container.reasoning_min_tokens,
                reasoning_max_tokens=container.reasoning_max_tokens,
                reasoning_token_scale=container.reasoning_token_scale,
                web_search=container.web_search,
                web_search_enabled=container.web_search_enabled,
                web_search_knowledge_enabled=container.web_search_knowledge_enabled,
                knowledge_temperature=container.knowledge_temperature,
                knowledge_max_tokens=container.knowledge_max_tokens,
                on_web_search_error=_log_web_search_error("chat"),
            )
            # Whether the turn is grounded is only known after retrieval, so the
            # policy that governs decoding is the one `prepare_turn` hands back.
            policy = turn.response_policy
            with log_duration(logger, "llm.chat", component="llm"):
                messages = await _with_optional_reasoning(container, turn, "chat")
                options = policy.options(container.llm.temperature, container.llm.max_tokens)
                response_text = await container.llm.chat(messages, options=options)
                # Three outcomes, not two. Some sentences flagged: keep what
                # survived. Every sentence flagged: ask the model once more,
                # telling it what happened -- this is the case the filter
                # exists for, and emitting the raw text here handed the user
                # back the very sentence BM25 had just caught.
                #
                # Only if the retry repeats too does the raw text stand. That
                # last resort is still the model's own words rather than a
                # fixed canned phrase, which would itself join the BM25 corpus
                # and, being short and generic, start colliding with future
                # turns and spamming itself back in an ever-tightening loop.
                kept = _filter_repeated_response(container, response_text)
                if not kept.strip():
                    kept = await _regenerate_without_repetition(
                        container, messages, options, "chat"
                    )
                response_text = kept.strip() or response_text
                # The prompt asked for no closing question; this is what makes
                # it true. On these weights the instruction held two times in
                # three, and the third is what the user actually noticed.
                if turn.forbids_questions:
                    response_text = strip_trailing_question(response_text)

        reading, state = await _emotion_state(container.emotion, response_text, turn.retrieved_docs)
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
        agent_vad=agent_vad,
        response_policy=ResponsePolicyResponse.from_policy(policy) if policy.is_active else None,
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
                # Gate covers retrieval too, not just the stream: RAG takes long
                # enough that a background generation slipping in during it would
                # still land in front of this turn's first token.
                async with container.llm_gate.foreground():
                    policy = derive_response_policy(body.user_vad, container.llm.max_tokens)
                    turn = await prepare_turn(
                        body.query, container.memory, container.rag, container.memory_recent_limit,
                        history_char_budget=container.memory_recent_char_budget,
                        on_rag_error=_log_rag_error("chat_stream"),
                        response_policy=policy,
                        reasoning_enabled=container.reasoning_enabled,
                        reasoning_activation_threshold=container.reasoning_activation_threshold,
                        reasoning_min_tokens=container.reasoning_min_tokens,
                        reasoning_max_tokens=container.reasoning_max_tokens,
                        reasoning_token_scale=container.reasoning_token_scale,
                        web_search=container.web_search,
                        web_search_enabled=container.web_search_enabled,
                        web_search_knowledge_enabled=container.web_search_knowledge_enabled,
                        knowledge_temperature=container.knowledge_temperature,
                        knowledge_max_tokens=container.knowledge_max_tokens,
                        on_web_search_error=_log_web_search_error("chat_stream"),
                    )
                    policy = turn.response_policy
                    docs = turn.retrieved_docs

                    stream_start_logged = False
                    messages = await _with_optional_reasoning(container, turn, "chat_stream")
                    pending = ""
                    raw_response = ""
                    # The buffered path can inspect a finished reply; a stream
                    # has to decide on each sentence before knowing whether it
                    # is the last, so a question is held one sentence rather
                    # than emitted. Costs one sentence of latency, and only on
                    # a turn that forbids questions and whose reply has one.
                    suppressor = TrailingQuestionSuppressor() if turn.forbids_questions else None
                    options = policy.options(container.llm.temperature, container.llm.max_tokens)
                    async for raw_chunk in container.llm.stream_chat(
                        messages, options=options,
                    ):
                        pending += raw_chunk
                        completed, pending = take_complete_sentences(pending)
                        for sentence in completed:
                            raw_response += sentence
                            released = suppressor.feed(sentence) if suppressor else [sentence]
                            for candidate in released:
                                chunk = _filter_repeated_response(container, candidate)
                                if not chunk:
                                    continue
                                if not stream_start_logged:
                                    logger.debug("llm.stream_chat | first token received", extra={"component": "llm"})
                                    stream_start_logged = True
                                if full_response and not chunk[0].isspace():
                                    chunk = " " + chunk
                                full_response += chunk
                                event = ChatStreamDeltaEvent(turn_id=turn_id, content=chunk)
                                yield f"data: {event.model_dump_json()}\n\n"

                    # The unterminated remainder is the last thing generated, so
                    # it goes through the suppressor too before the flush that
                    # drops anything still being held.
                    if suppressor is not None:
                        released_tail = suppressor.feed(pending) if pending.strip() else []
                        released_tail += suppressor.flush()
                    else:
                        released_tail = [pending] if pending.strip() else []

                    emitted_tail = False
                    for candidate in released_tail:
                        tail = _filter_repeated_response(container, candidate)
                        if not tail:
                            continue
                        if full_response and not tail[0].isspace():
                            tail = " " + tail
                        full_response += tail
                        emitted_tail = True
                        event = ChatStreamDeltaEvent(turn_id=turn_id, content=tail)
                        yield f"data: {event.model_dump_json()}\n\n"

                    if not emitted_tail and not full_response:
                        # Every sentence got flagged as repetitive. Nothing has
                        # reached the client yet, so the turn can still be
                        # rewritten -- ask once more with the reason, and only
                        # keep the raw text if that comes back repetitive too.
                        #
                        # The retry is generated whole rather than streamed. It
                        # would mean running this loop's machinery a second
                        # time for a path the logs put at a few percent of
                        # turns, and the fallback it replaces emitted its text
                        # in one delta anyway.
                        retry = await _regenerate_without_repetition(
                            container, messages, options, "chat_stream"
                        )
                        # Last resort stays the model's own words, not a fixed
                        # canned phrase, which would itself join the BM25 corpus
                        # and spam back in later turns. The canned line is only
                        # for a turn that generated nothing at all.
                        full_response = (
                            retry.strip()
                            or (raw_response + pending).strip()
                            or "Mình hiểu rồi."
                        )
                        event = ChatStreamDeltaEvent(turn_id=turn_id, content=full_response)
                        yield f"data: {event.model_dump_json()}\n\n"

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
