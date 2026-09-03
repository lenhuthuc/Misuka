"""Node 4: judge the draft before it becomes the reply.

The rules are in `brain.graph.guard`; this node is only where the verdict is
turned into a decision about the turn. Two things it must respect:

`emitted` -- on the streaming endpoint the guard has already vetted every
sentence on its way out, and words that have been spoken cannot be recalled. So
a regeneration is possible only when nothing reached the client. That is not a
weaker guard: the per-sentence pass in `generate` uses this same `ReplyGuard`,
so the same rules applied; what changes is the remedy available afterwards.

the retry budget -- one regeneration, then the best available text stands. A
reply that breaks a style rule is a better failure than silence, and the one
rule that does not degrade to "say it anyway" is violence, which is why it is
the pattern that never regenerates and always substitutes.
"""
from __future__ import annotations

import logging

from langgraph.config import get_stream_writer

from brain.graph.deps import GraphDeps
from brain.graph.guard import (
    PASS,
    REGENERATE,
    SAFE_LINE,
    SAFE_REPLY,
    GuardVerdict,
    ReplyGuard,
)
from brain.graph.state import MitsukaState

logger = logging.getLogger(__name__)


def make_guard(deps: GraphDeps):
    async def guard_node(state: MitsukaState) -> dict:
        guard: ReplyGuard | None = state.get("guard")
        draft = state.get("draft_reply", "")
        flags = list(state.get("flags") or [])
        writer = get_stream_writer()

        # Violence caught mid-stream: `generate` stopped at the sentence and
        # never released it. Say the safe line instead -- and push it to a
        # streaming client, which is otherwise left with a reply that just
        # stops.
        if guard is not None and guard.aborted:
            writer({"delta": SAFE_REPLY})
            logger.warning("graph.guard | stream aborted on violence, safe line sent")
            return {
                "draft_reply": SAFE_REPLY,
                "guard_action": SAFE_LINE,
                "flags": flags,
                "emitted": True,
            }

        # `guard` is None only if generation never ran (a backend raised before
        # building one), in which case there is nothing to judge.
        verdict = (
            guard.check(draft)
            if guard is not None
            else GuardVerdict(text=draft, action=PASS, flags=list(flags))
        )
        merged = flags + [f for f in verdict.flags if f not in flags]

        if verdict.action == SAFE_LINE:
            # Only reachable on the buffered path, where nothing was streamed.
            if state.get("emitted"):
                writer({"delta": SAFE_REPLY})
            logger.warning("graph.guard | violence in reply, substituting safe line")
            return {
                "draft_reply": verdict.text,
                "guard_action": SAFE_LINE,
                "flags": merged,
                "emitted": True,
            }

        if verdict.action == REGENERATE:
            spent = state.get("regenerate_count", 0)
            can_retry = (
                spent < deps.config.max_regenerations and not state.get("emitted", False)
            )
            if can_retry:
                logger.info("graph.guard | regenerating once | flags=%s", merged)
                return {
                    "guard_action": REGENERATE,
                    "flags": merged,
                    "regenerate_count": spent + 1,
                    "draft_reply": verdict.text,
                }
            # Out of retries, or the words are already out. The model's own
            # text stands rather than a canned line -- a fixed phrase would
            # join the BM25 corpus and start colliding with future turns.
            logger.info(
                "graph.guard | keeping flagged reply | flags=%s emitted=%s spent=%d",
                merged, state.get("emitted", False), spent,
            )
            kept = verdict.text or state.get("raw_reply", "")
            emitted = state.get("emitted", False)
            # A streaming attempt whose entire reply was held by the guard has
            # not sent anything yet. Once the retry budget is exhausted, emit
            # the model's own fallback text so the SSE client and final state
            # cannot disagree about what Mitsuka said.
            if state.get("streaming") and not emitted and kept:
                writer({"delta": kept})
                emitted = True
            return {
                "guard_action": "kept_flagged",
                "flags": merged,
                "draft_reply": kept,
                "emitted": emitted,
            }

        return {"draft_reply": verdict.text, "guard_action": PASS, "flags": merged}

    return guard_node


def after_guard(state: MitsukaState) -> str:
    """Conditional edge out of `guard`: retry on the same route, or move on."""
    if state.get("guard_action") != REGENERATE:
        return "finalize"
    return "generate_cloud" if state.get("route") == "cloud" else "generate_local"
