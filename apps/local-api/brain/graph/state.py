"""The graph's state: everything one spoken turn carries between nodes.

The eight fields the design calls for -- session_id, user_text, history,
rolling_summary, draft_reply, final_reply, route, flags -- are the contract and
come first. The block after them is what this app's turn actually needs on top
of that: retrieval results, the VAD-derived decoding policy, the deliberation
decision. They live here rather than in a side channel because a LangGraph node
gets exactly one thing to read and write, and the moment some of a turn's data
travels outside the state, the graph stops being the description of the turn.

State belongs to the GRAPH, not to a model. Nothing in here is backend-specific
except `route` itself, which is recorded so the metrics line can say which
backend answered -- switching mid-conversation must not cost Mitsuka her memory.
"""
from __future__ import annotations

from typing import Any, TypedDict

from brain.reasoning_policy import ReasoningPolicy
from brain.response_policy import ResponsePolicy
from schemas.vad import VADScores


class Turn(TypedDict, total=False):
    """One exchange as the prompt and the summariser see it."""

    role: str      # "user" | "assistant"
    content: str


class MitsukaState(TypedDict, total=False):
    # ── The contract ─────────────────────────────────────────────────────────
    session_id: str
    user_text: str
    history: list[Turn]
    rolling_summary: str
    # What the model produced, before `guard` has passed judgement on it.
    draft_reply: str
    # What goes to TTS. Only `finalize` writes this.
    final_reply: str
    route: str          # "cloud" | "local"
    flags: list[str]

    # ── What this app's turn needs on top of the contract ────────────────────
    turn_id: str
    user_vad: VADScores | None
    # Assembled by `generate`; kept on the state because `guard` regenerates
    # from the same prompt rather than rebuilding it.
    messages: list[dict[str, str]]
    generated_queries: list[str]
    retrieved_docs: list[dict[str, Any]]
    web_results: list[dict[str, Any]]
    reasoning: ReasoningPolicy
    response_policy: ResponsePolicy
    forbids_questions: bool

    # The turn's `brain.graph.guard.ReplyGuard`. It is a live object rather
    # than data because the streaming path has to consult the same rules
    # sentence by sentence that the `guard` node applies to the whole reply,
    # and two objects would be two sets of rules. Nothing checkpoints this
    # state -- durability is `session_state` in brain.db -- so it does not need
    # to be serialisable.
    guard: Any
    # Everything the backend produced, before the guard touched it. Kept apart
    # from `draft_reply` so a regeneration can tell "the model said nothing"
    # from "the guard removed all of it".
    raw_reply: str
    # Whether this invocation has a client consuming the custom stream. The
    # graph always calls the stream writer, but under `ainvoke` it is a no-op;
    # without this bit, buffered output was incorrectly treated as irrevocably
    # emitted and could never take the guard's regeneration edge.
    streaming: bool
    # True once any text has reached the client. A streamed turn cannot be
    # regenerated after this, because the words are already spoken.
    emitted: bool
    regenerate_count: int
    guard_action: str

    # ── Measurement ──────────────────────────────────────────────────────────
    # Accumulated across nodes and written out once by `update_memory`; see
    # `brain.graph.metrics`.
    metrics: dict[str, Any]


def new_state(
    user_text: str,
    session_id: str,
    turn_id: str,
    user_vad: VADScores | None = None,
) -> MitsukaState:
    """A state with every field present, so no node has to guess a default."""
    return MitsukaState(
        session_id=session_id,
        user_text=user_text,
        history=[],
        rolling_summary="",
        draft_reply="",
        final_reply="",
        route="",
        flags=[],
        turn_id=turn_id,
        user_vad=user_vad,
        messages=[],
        generated_queries=[],
        retrieved_docs=[],
        web_results=[],
        reasoning=ReasoningPolicy(),
        response_policy=ResponsePolicy(),
        forbids_questions=False,
        guard=None,
        raw_reply="",
        streaming=False,
        emitted=False,
        regenerate_count=0,
        guard_action="pass",
        metrics={},
    )
