"""Node 1: load this session's history and rolling summary.

Reads from `brain.db` -- the same file the transcript, RAG's source rows and the
BM25 index already live in. A LangGraph checkpointer would have been fewer lines
here and a second copy of the conversation everywhere else: retrieval and the
repetition filter read the transcript table directly, so a separate checkpoint
store would leave them looking at an older conversation than the graph's.
"""
from __future__ import annotations

import logging
import time

from brain.graph.deps import GraphDeps
from brain.graph.state import MitsukaState, Turn

logger = logging.getLogger(__name__)


def make_load_context(deps: GraphDeps):
    async def load_context(state: MitsukaState) -> dict:
        session_id = state["session_id"]

        # Two rows per turn (user + assistant), so the window holds
        # `history_turns` exchanges rather than half that many.
        rows = await deps.memory.get_recent(
            deps.config.history_turns * 2, session_id=session_id
        )
        history: list[Turn] = [
            {"role": str(row["role"]), "content": str(row["content"])} for row in rows
        ]
        rolling_summary = await deps.memory.get_rolling_summary(session_id)

        logger.debug(
            "graph.load_context | session=%s history=%d summary_chars=%d",
            session_id, len(history), len(rolling_summary),
        )
        return {
            "history": history,
            "rolling_summary": rolling_summary,
            # Latency is measured from here: the turn's clock starts when the
            # graph starts, not when the model is finally called.
            "metrics": {**state.get("metrics", {}), "started_at": time.perf_counter()},
        }

    return load_context
