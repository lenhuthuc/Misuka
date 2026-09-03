"""The LangGraph conversation layer: `brain.graph`.

Entry points are `MitsukaGraph` (a compiled turn graph plus the buffered and
streaming ways to run it) and `GraphDeps`/`GraphConfig` (what it runs against
and what it tunes). Everything else is a node.
"""
from __future__ import annotations

from brain.graph.deps import GraphConfig, GraphDeps
from brain.graph.graph import MitsukaGraph, build_graph
from brain.graph.state import MitsukaState, new_state

__all__ = [
    "GraphConfig",
    "GraphDeps",
    "MitsukaGraph",
    "MitsukaState",
    "build_graph",
    "new_state",
]
