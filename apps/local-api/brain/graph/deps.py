"""What the graph's nodes are allowed to reach, and every knob they read.

LangGraph nodes are functions of state, so services have to arrive some other
way. They arrive here: each node is built by a factory that closes over one
`GraphDeps`, which makes the dependency list of every node explicit and lets a
caller build a graph over fakes without patching imports.

`GraphConfig` is the "all tunables in one place" half. The values still *live*
in `brain/config.py` (one Settings class, env-overridable) -- this is the subset
the graph reads, copied once at construction, so a node never reaches for a
global.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from brain.graph.backends import BackendRegistry
from brain.graph.guard import GuardConfig
from brain.graph.metrics import TurnMetrics

if TYPE_CHECKING:
    from brain.llm_service import LLMService
    from brain.memory_service import MemoryService
    from brain.rag_service import RAGService
    from brain.web_search_service import WebSearchService
    from core.llm_priority import LLMPriorityGate
    from core.tasks import BackgroundTaskRegistry


@dataclass
class GraphConfig:
    """Everything the graph tunes, mirrored from `brain.config.Settings`."""

    # -- History and summarisation ------------------------------------------
    # Turns kept verbatim in the graph's `history` before the oldest are folded
    # into `rolling_summary`. Distinct from `memory_recent_limit`, which is how
    # many rows the *prompt* carries -- that one is a prefill/latency budget and
    # is deliberately smaller.
    history_turns: int = 12
    summary_max_tokens: int = 160
    # How long the summariser waits for the reply to finish being spoken before
    # giving up on the signal and running anyway. Mirrors `index_defer_timeout`.
    summary_defer_timeout: float = 90.0
    memory_recent_limit: int = 6
    memory_recent_char_budget: int = 1600

    # -- Decoding ------------------------------------------------------------
    temperature: float = 0.65
    max_tokens: int = 512

    # -- Deliberation (local backend only) -----------------------------------
    reasoning_enabled: bool = True
    reasoning_activation_threshold: float = 0.50
    reasoning_min_tokens: int = 64
    reasoning_max_tokens: int = 192
    reasoning_token_scale: float = 0.45

    # -- Retrieval / web -----------------------------------------------------
    web_search_enabled: bool = True
    web_search_knowledge_enabled: bool = True
    knowledge_temperature: float = 0.30
    knowledge_max_tokens: int = 480

    # -- Guard ---------------------------------------------------------------
    guard: GuardConfig = field(default_factory=GuardConfig)
    # One retry, matching the existing repetition-retry path. A second would
    # double worst-case latency for a case the logs put at a few percent.
    max_regenerations: int = 1


@dataclass
class GraphDeps:
    """Services one graph instance runs against."""

    memory: "MemoryService"
    rag: "RAGService"
    backends: BackendRegistry
    metrics: TurnMetrics
    config: GraphConfig
    # Ollama-only: the hidden deliberation pass runs on a stock qwen3:1.7b and
    # has no cloud counterpart, by design (`CloudBackend.supports_reasoning`).
    llm: "LLMService | None" = None
    web_search: "WebSearchService | None" = None
    # Background summarisation waits behind the user's turn through this.
    gate: "LLMPriorityGate | None" = None
    # Where the summariser is spawned. None means "run it inline" -- correct for
    # the demo script and tests, which have no request to stay out of the way of.
    tasks: "BackgroundTaskRegistry | None" = None
