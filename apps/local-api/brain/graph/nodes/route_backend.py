"""Node 2: decide which backend answers, and record why.

The decision is a node; the *branch* is `select_backend`, a conditional edge.
That split is deliberate -- putting the branch inside the node would hide the
fallback in an `if`, and the whole point of routing through the graph is that
"which model answered, and why" is visible in the graph's shape and in
`state["route"]` rather than buried in control flow.

Cloud is the default. It gives way to local on exactly three conditions:

  (a) the cloud call failed to connect          -> remembered by CloudAvailability
  (b) today's free-tier allowance is spent      -> remembered by CloudAvailability
  (c) GEMINI_API_KEY is not configured          -> known without trying

(a) and (b) can only be learned by having tried, so they are read from the
circuit breaker rather than probed here -- a health-check ping before every turn
would spend a request and a round trip to ask a question the last failure has
already answered.
"""
from __future__ import annotations

import logging

from brain.graph.deps import GraphDeps
from brain.graph.state import MitsukaState

logger = logging.getLogger(__name__)

CLOUD_NODE = "generate_cloud"
LOCAL_NODE = "generate_local"


def make_route_backend(deps: GraphDeps):
    async def route_backend(state: MitsukaState) -> dict:
        route, reason = deps.backends.choose()
        if reason:
            logger.info("graph.route | route=%s reason=%s", route, reason)
        return {
            "route": route,
            "metrics": {**state.get("metrics", {}), "fallback_reason": reason},
        }

    return route_backend


def select_backend(state: MitsukaState) -> str:
    """Conditional edge out of `route_backend`."""
    return CLOUD_NODE if state.get("route") == "cloud" else LOCAL_NODE


def after_cloud(state: MitsukaState) -> str:
    """Conditional edge out of `generate_cloud`.

    A cloud attempt that failed on a routing-relevant error leaves `route` set
    back to "local", and the turn re-enters generation on the local backend
    instead of failing. The retry is an edge rather than a try/except around a
    second call so that the fallback is one hop in the graph, and so the local
    attempt runs the identical node the local-first path runs.
    """
    return LOCAL_NODE if state.get("route") == "local" else "guard"
