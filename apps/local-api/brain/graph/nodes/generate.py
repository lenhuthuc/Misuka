"""Node 3: produce `draft_reply` on the chosen backend.

One node body serves four jobs that used to be two code paths:

  * buffered and streaming. The node always streams from the backend and always
    publishes each finished sentence through LangGraph's custom stream writer.
    Under `ainvoke` that writer is a no-op, so `/v1/chat` gets the accumulated
    string and `/v1/chat/stream` gets the same sentences as they land -- from
    the same lines of code. The previous graph attempt was removed because the
    two endpoints had drifted into separate implementations; this is the part
    that makes one implementation possible.
  * cloud and local. Which backend is a lookup, not a branch: the prompt is
    assembled once, and `ChatBackend.prepare` adapts it to how that backend
    receives a persona.
  * first attempt and the guard's one retry, distinguished only by the note
    appended to the prompt.

Sentence granularity is not a stylistic choice either. The guard has to see a
complete sentence to judge it, and a streamed reply has to be judged before it
is spoken, so the stream is cut on sentence boundaries and released one sentence
behind the model. That costs a sentence of latency and is what makes moderating
a stream possible at all.
"""
from __future__ import annotations

import logging
import time

from langgraph.config import get_stream_writer

from application.conversation_turn import prepare_turn
from brain.bm25_repetition import take_complete_sentences
from brain.graph.backends import BackendError, BackendUnavailable, DailyQuotaExceeded
from brain.graph.backends.base import GenerationOptions, Usage
from brain.graph.deps import GraphDeps
from brain.graph.guard import ReplyGuard
from brain.graph.state import MitsukaState
from brain.reasoning_policy import inject_reasoning_note
from brain.response_policy import derive_response_policy

logger = logging.getLogger(__name__)

# Sent only on the guard's retry. It states the finding rather than becoming a
# standing rule: the standing rules are read on every turn, and this is true on
# a few percent of them.
_RETRY_NOTES = {
    "repetition": (
        "Câu trả lời bạn vừa viết lặp lại gần như nguyên văn điều bạn đã nói "
        "trước đó, nên nó bị bỏ. Viết lại bằng nội dung khác hẳn: trả lời thẳng "
        "vào điều người dùng vừa nói, không hỏi lại, không lặp lời mời cũ."
    ),
    "empty": (
        "Câu trả lời vừa rồi trống. Hãy trả lời ngắn gọn 1-3 câu, đi thẳng vào "
        "điều người dùng vừa nói."
    ),
    "too_long": (
        "Câu trả lời vừa rồi quá dài. Viết lại thật ngắn, tối đa 3 câu, giống "
        "lời nói trực tiếp."
    ),
    "self_ending": (
        "Câu trả lời vừa rồi tự kết thúc cuộc trò chuyện. Viết lại mà không "
        "chào tạm biệt và không khép lại cuộc nói chuyện."
    ),
}
_DEFAULT_RETRY_NOTE = _RETRY_NOTES["repetition"]


def retry_note_for(flags: list[str]) -> str:
    """The note that names what actually went wrong, not a generic scolding."""
    for flag in flags:
        if flag in _RETRY_NOTES:
            return _RETRY_NOTES[flag]
    return _DEFAULT_RETRY_NOTE


def with_retry_note(messages: list[dict[str, str]], note: str) -> list[dict[str, str]] | None:
    """Same messages, plus the note in the only position that works.

    The note has to land immediately before the user's question. After the user
    message the model reads it as the user's own words; at index 0 it silently
    replaces the Modelfile persona on the local backend. Both were measured --
    see the layout rule at the top of `brain/nodes/generate.py`. A turn with no
    history has no such position, so it keeps its first reply rather than
    trading the persona for a retry.
    """
    if len(messages) < 2:
        return None
    head, user = list(messages[:-1]), messages[-1]
    if head[-1]["role"] == "system":
        head[-1] = {**head[-1], "content": f"{head[-1]['content']}\n\n{note}"}
    else:
        head.append({"role": "system", "content": note})
    return [*head, user]


async def _build_prompt(state: MitsukaState, deps: GraphDeps) -> dict:
    """Retrieval, web search, VAD policy and message assembly for this turn.

    Delegated to `application.conversation_turn.prepare_turn`, which is the
    existing single source of that policy. The graph does not reimplement it.
    """
    policy = derive_response_policy(state.get("user_vad"), deps.config.max_tokens)
    turn = await prepare_turn(
        state["user_text"],
        deps.memory,
        deps.rag,
        deps.config.memory_recent_limit,
        history_char_budget=deps.config.memory_recent_char_budget,
        on_rag_error=lambda exc: logger.exception("graph.generate | RAG failed, skipping"),
        response_policy=policy,
        reasoning_enabled=deps.config.reasoning_enabled,
        reasoning_activation_threshold=deps.config.reasoning_activation_threshold,
        reasoning_min_tokens=deps.config.reasoning_min_tokens,
        reasoning_max_tokens=deps.config.reasoning_max_tokens,
        reasoning_token_scale=deps.config.reasoning_token_scale,
        web_search=deps.web_search,
        web_search_enabled=deps.config.web_search_enabled,
        web_search_knowledge_enabled=deps.config.web_search_knowledge_enabled,
        knowledge_temperature=deps.config.knowledge_temperature,
        knowledge_max_tokens=deps.config.knowledge_max_tokens,
        on_web_search_error=lambda exc: logger.exception(
            "graph.generate | web search failed, skipping"
        ),
        session_id=state["session_id"],
        rolling_summary=state.get("rolling_summary", ""),
    )
    return {
        "messages": turn.messages,
        "generated_queries": turn.generated_queries,
        "retrieved_docs": list(turn.retrieved_docs),
        "web_results": turn.web_results,
        "reasoning": turn.reasoning,
        "response_policy": turn.response_policy,
        "forbids_questions": turn.forbids_questions,
    }


def _previous_assistant(history: list[dict[str, str]]) -> str:
    for row in reversed(history):
        if row.get("role") == "assistant":
            return str(row.get("content", ""))
    return ""


def make_generate(deps: GraphDeps, route: str):
    """Build the generation node bound to one backend.

    `route` is bound at construction so `generate_cloud` and `generate_local`
    are two nodes in the graph -- the fallback is then a visible edge -- while
    remaining one implementation.
    """

    async def generate(state: MitsukaState) -> dict:
        backend = deps.backends.get(route)
        update: dict = {"route": route}

        # First pass only: retrieval, policy and prompt. A regeneration reuses
        # the prompt it already paid for and changes only the note on the end.
        if not state.get("messages"):
            update |= await _build_prompt(state, deps)
        messages = list(update.get("messages") or state["messages"])
        reasoning = update.get("reasoning", state.get("reasoning"))
        policy = update.get("response_policy", state.get("response_policy"))
        forbids_questions = update.get(
            "forbids_questions", state.get("forbids_questions", False)
        )

        # Deliberation is local-only: Gemini does not need a 1.7B's crutch, and
        # the pass would spend an extra free-tier request per turn to add one.
        if (
            backend.supports_reasoning
            and deps.llm is not None
            and reasoning is not None
            and reasoning.enabled
        ):
            try:
                note = await deps.llm.reason(messages, max_tokens=reasoning.token_budget)
                messages = inject_reasoning_note(messages, note)
            except Exception:
                # Never a turn dependency: a missing reasoning model must not
                # turn an answerable question into a 5xx.
                logger.exception("graph.generate | reasoning failed, answering directly")

        if state.get("regenerate_count", 0):
            flags = list(state.get("flags") or [])
            retried = with_retry_note(messages, retry_note_for(flags))
            if retried is None:
                logger.info("graph.generate | nowhere to place retry note, keeping reply")
                return {**update, "draft_reply": state.get("raw_reply", "")}
            messages = retried

        # A fresh guard per attempt, never the previous one. Its trailing-
        # question suppressor holds at most one sentence between calls, and a
        # reused guard would open the retry still holding a sentence from the
        # reply that was just thrown away. Prior flags travel separately, so
        # the metrics line still reports why the retry happened.
        prior_flags = list(state.get("flags") or [])
        guard = ReplyGuard(
            user_text=state["user_text"],
            previous_assistant=_previous_assistant(state.get("history") or []),
            forbids_questions=forbids_questions,
            config=deps.config.guard,
            bm25_filter=getattr(deps.memory, "filter_assistant_response", None),
        )
        update["guard"] = guard

        options = GenerationOptions(
            temperature=(
                policy.temperature if policy and policy.temperature is not None
                else deps.config.temperature
            ),
            max_tokens=(
                policy.max_tokens if policy and policy.max_tokens is not None
                else deps.config.max_tokens
            ),
        )

        writer = get_stream_writer()
        usage = Usage()
        started = time.perf_counter()
        raw = ""
        released: list[str] = []
        pending = ""
        emitted = state.get("emitted", False)

        try:
            async for chunk in backend.astream(messages, options, usage):
                raw += chunk
                pending += chunk
                completed, pending = take_complete_sentences(pending)
                for sentence in completed:
                    for safe in guard.feed(sentence):
                        released.append(safe)
                        emitted = emitted or bool(state.get("streaming"))
                        writer({"delta": safe})
                if guard.aborted:
                    break

            if not guard.aborted:
                # The unterminated tail is still a sentence the guard must see.
                if pending.strip():
                    for safe in guard.feed(pending):
                        released.append(safe)
                        emitted = emitted or bool(state.get("streaming"))
                        writer({"delta": safe})
                for safe in guard.flush():
                    released.append(safe)
                    emitted = emitted or bool(state.get("streaming"))
                    writer({"delta": safe})

        except (BackendUnavailable, DailyQuotaExceeded) as exc:
            # The two conditions `route_backend` cannot know in advance. Record
            # them so the next turn routes without paying this failure again,
            # and hand the turn to the local backend via `after_cloud`.
            if route == "cloud":
                if isinstance(exc, DailyQuotaExceeded):
                    deps.backends.availability.record_daily_quota(str(exc))
                else:
                    deps.backends.availability.record_transient(str(exc))
                logger.warning("graph.generate | cloud failed, falling back to local: %s", exc)
                return {
                    **update,
                    "route": "local",
                    # A cloud attempt that emitted nothing leaves no trace for
                    # the user; one that already streamed text cannot be redone,
                    # and `guard` will take what there is.
                    "emitted": emitted,
                    "raw_reply": raw,
                    "draft_reply": "".join(released),
                    "metrics": {
                        **state.get("metrics", {}),
                        "fallback_reason": type(exc).__name__,
                    },
                }
            raise
        except BackendError:
            logger.exception("graph.generate | backend error on route=%s", route)
            raise

        if route == "cloud":
            deps.backends.availability.record_success()

        latency_ms = (time.perf_counter() - started) * 1000
        metrics = {
            **state.get("metrics", {}),
            "model": getattr(backend, "model", ""),
            "input_tokens": state.get("metrics", {}).get("input_tokens", 0) + usage.input_tokens,
            "output_tokens": state.get("metrics", {}).get("output_tokens", 0) + usage.output_tokens,
            "generation_ms": state.get("metrics", {}).get("generation_ms", 0.0) + latency_ms,
        }

        logger.info(
            "graph.generate | route=%s chars=%d released=%d in_tok=%d out_tok=%d %.0fms",
            route, len(raw), len(released), usage.input_tokens, usage.output_tokens, latency_ms,
        )
        return {
            **update,
            "raw_reply": raw,
            "draft_reply": "".join(released),
            "emitted": emitted,
            "flags": prior_flags + [f for f in guard.flags if f not in prior_flags],
            "metrics": metrics,
        }

    return generate
