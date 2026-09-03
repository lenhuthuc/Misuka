"""Node 5: settle on the reply and append the exchange to `history`.

`final_reply` is written here and nowhere else. That is the whole job: every
other node may propose text, and a reader looking for "what did Mitsuka
actually say" has exactly one assignment to find.

The history written here is the graph's in-memory view, used by
`update_memory` to decide what has fallen out of the window. Durable storage of
the exchange is a separate step, and stays where it already was -- the
background task in `brain.background`, which also does the vector indexing and
the fact extraction.
"""
from __future__ import annotations

import logging

from brain.graph.deps import GraphDeps
from brain.graph.state import MitsukaState

logger = logging.getLogger(__name__)

# Only for a turn that produced no text at all -- not for a flagged reply,
# which keeps the model's own words. A canned line that shipped on every
# flagged turn would repeat itself into the BM25 corpus and start suppressing
# itself.
_NOTHING_GENERATED = "Mình nghe bạn nè."


def make_finalize(deps: GraphDeps):
    async def finalize(state: MitsukaState) -> dict:
        reply = (state.get("draft_reply") or "").strip()
        if not reply:
            reply = (state.get("raw_reply") or "").strip() or _NOTHING_GENERATED
            logger.info("graph.finalize | empty reply, using fallback line")

        history = list(state.get("history") or [])
        history.append({"role": "user", "content": state["user_text"]})
        history.append({"role": "assistant", "content": reply})

        return {"final_reply": reply, "history": history}

    return finalize
