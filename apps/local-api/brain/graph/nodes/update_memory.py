"""Node 6: fold what fell out of the window into the summary, and log the turn.

Two jobs, and the ordering between them matters. The metrics line is written
first and unconditionally, because it is the record of what the turn cost and a
summarisation failure must not be able to erase it.

Summarisation is *scheduled*, not awaited. Ollama serves one request at a time
per model and decode here is memory-bandwidth-bound, so a summary generated
inline would sit directly between the user's reply and their next sentence.
It is spawned as a background task that first waits on
`LLMPriorityGate.wait_until_spoken()` -- the same shape `brain.background` uses
for vector indexing. The conversation is not waiting on it: `rolling_summary`
covers messages that have already left the prompt, so being one turn late costs
nothing.

Worth stating plainly, because `core/llm_priority.py` says it in its own
docstring: this is the first background *LLM* call on the server since
`run_when_idle` and its interrupt channel were removed. `wait_until_spoken`
keeps the summary off the runner until the reply has been played, but nothing
abandons it if the user starts speaking again during it. The exposure is
bounded on purpose -- it fires only on the turns where the history window
actually overflows (once every `history_turns` exchanges, not every turn), and
`summary_max_tokens` caps the generation. If that turns out to be audible, the
fix is to bring back an abandon-on-activity gate, not to move this inline.
"""
from __future__ import annotations

import logging
import time

from brain.graph.deps import GraphDeps
from brain.graph.metrics import build_record
from brain.graph.state import MitsukaState

logger = logging.getLogger(__name__)

_SUMMARY_SYSTEM = (
    "Bạn là bộ nhớ tóm tắt của một trợ lý hội thoại. Gộp tóm tắt cũ và các lượt "
    "mới thành một đoạn tiếng Việt ngắn, tối đa 5 câu, chỉ giữ thông tin còn "
    "hữu ích về sau: tên, sở thích, sự kiện, việc đang làm dở, cảm xúc kéo dài. "
    "Bỏ lời chào, lời cảm ơn và chi tiết vụn vặt. Chỉ trả về đoạn tóm tắt."
)


def _render_span(rows: list[dict]) -> str:
    speakers = {"user": "Người dùng", "assistant": "Mitsuka"}
    return "\n".join(
        f"{speakers.get(str(row['role']), str(row['role']))}: {row['content']}"
        for row in rows
    )


async def _summarize(deps: GraphDeps, previous: str, rows: list[dict]) -> str:
    """Ask the local model to merge the old summary with the fallen-out span.

    Deliberately the chat model rather than the stock `reasoning_model`: this
    runs on a box where the 1.1GB `mitsuka-ft` is already resident with a 30m
    keep-alive, and pulling a second model in for a background chore would evict
    it and hand the user's next turn a cold load. The persona bleeding into the
    wording of a summary is harmless -- it is read back as context, not spoken.
    """
    if deps.llm is None:
        # No model to summarise with: keep the previous summary rather than
        # replacing it with a concatenation that would grow without bound.
        return previous
    prompt = (
        f"Tóm tắt hiện có:\n{previous or '(chưa có)'}\n\n"
        f"Các lượt mới cần gộp vào:\n{_render_span(rows)}\n\nTóm tắt gộp:"
    )
    summary = await deps.llm.generate(prompt, system=_SUMMARY_SYSTEM)
    return summary.strip()


def make_update_memory(deps: GraphDeps):
    async def update_memory(state: MitsukaState) -> dict:
        metrics = state.get("metrics", {})
        started = metrics.get("started_at")
        latency_ms = (time.perf_counter() - started) * 1000 if started else 0.0

        deps.metrics.write(
            build_record(
                session_id=state.get("session_id", ""),
                turn_id=state.get("turn_id", ""),
                route=state.get("route", ""),
                model=metrics.get("model", ""),
                input_tokens=metrics.get("input_tokens", 0),
                output_tokens=metrics.get("output_tokens", 0),
                latency_ms=latency_ms,
                flags=list(state.get("flags") or []),
                guard_action=state.get("guard_action", "pass"),
                regenerated=bool(state.get("regenerate_count", 0)),
                fallback_reason=metrics.get("fallback_reason", ""),
                streamed=bool(state.get("emitted")),
            )
        )

        await _schedule_summary(deps, state)
        return {}

    return update_memory


async def _schedule_summary(deps: GraphDeps, state: MitsukaState) -> None:
    """Queue a summary refresh if messages have dropped out of the window."""
    session_id = state.get("session_id", "default")
    window = deps.config.history_turns * 2

    # The window is defined by the same query `load_context` uses, so "oldest
    # row still carried verbatim" is read rather than inferred.
    rows = await deps.memory.get_recent(window, session_id=session_id)
    if len(rows) < window:
        return  # nothing has fallen out yet
    oldest_kept_id = int(rows[0]["id"])

    previous, watermark = await deps.memory.get_session_state(session_id)
    fallen_out = await deps.memory.get_messages_between(
        watermark, oldest_kept_id, session_id=session_id
    )
    if not fallen_out:
        return

    new_watermark = int(fallen_out[-1]["id"])

    async def run() -> None:
        # Stay off the runner until the reply the user is listening to has
        # actually been played. Same contract as `brain.background`: a client
        # that never reports its playback queue draining must not strand the
        # summary forever, so the timeout runs it late rather than dropping it.
        if deps.gate is not None:
            await deps.gate.wait_until_spoken(timeout=deps.config.summary_defer_timeout)
        try:
            summary = await _summarize(deps, previous, fallen_out)
        except Exception:
            logger.exception("graph.update_memory | summarisation failed")
            return
        if not summary:
            return
        await deps.memory.set_rolling_summary(
            summary, session_id=session_id, summarized_through_id=new_watermark
        )
        logger.info(
            "graph.update_memory | summary refreshed | session=%s folded=%d chars=%d",
            session_id, len(fallen_out), len(summary),
        )

    if deps.tasks is not None:
        deps.tasks.spawn(run(), name="graph.summarize")
    else:
        # No registry (the demo script, tests): run inline. Correct, just not
        # off the critical path -- and those callers have no critical path.
        await run()
