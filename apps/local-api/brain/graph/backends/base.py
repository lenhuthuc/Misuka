"""The interface both model backends sit behind.

Two things are deliberately part of this interface rather than left to callers:

`prepare` -- how the persona reaches the model. The text is identical for both
backends (`brain.graph.prompts` enforces that), but the delivery is not: Ollama
injects the Modelfile SYSTEM and is *destroyed* by a system message at index 0,
Gemini has no Modelfile and must be sent one. Callers assemble the turn's
messages once, and each backend adapts them on the way out.

The error taxonomy -- routing is a decision about *why* a call failed. "Gemini
is unreachable" and "today's free quota is gone" both mean fall back to local,
but they differ in how long that stays true, and an ordinary 500 means neither.
Squashing them into one exception is what turns a five-minute network blip into
an all-day local fallback.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass


class BackendError(RuntimeError):
    """Any failure from a model backend."""


class BackendUnavailable(BackendError):
    """The backend could not be reached: no network, DNS, TLS, timeout, no key.

    Transient by assumption. The circuit breaker retries this one soon.
    """


class DailyQuotaExceeded(BackendError):
    """The backend's per-day free-tier allowance is spent.

    Distinct from a per-minute 429, which is transient and worth retrying in
    seconds; this one does not clear until the quota window rolls over.
    """


@dataclass
class Usage:
    """Token counts for one call, as the backend reported them.

    Zeros mean "not reported", not "free": Ollama omits the counts on some
    responses. The metrics line keeps them zero rather than estimating, so a
    number in `logs/turns.jsonl` is always something a backend actually said.
    """

    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class Generation:
    """A complete reply plus what it cost."""

    text: str
    usage: Usage


@dataclass(frozen=True)
class GenerationOptions:
    """Decoding controls for one turn, backend-neutral."""

    temperature: float
    max_tokens: int


class ChatBackend(ABC):
    """One way of turning a turn's messages into Mitsuka's next sentence."""

    #: "cloud" | "local" -- the value that lands in `state["route"]`.
    name: str

    #: Whether the hidden deliberation pass is worth running before this
    #: backend answers. False for cloud: Gemini 2.5 Flash-Lite does not need a
    #: 1.7B's crutch, and the pass would add a round trip and burn free-tier
    #: requests to tell a stronger model what it already worked out.
    supports_reasoning: bool

    @abstractmethod
    def prepare(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        """Adapt the turn's messages to this backend's persona delivery."""

    @abstractmethod
    async def agenerate(
        self, messages: list[dict[str, str]], options: GenerationOptions
    ) -> Generation:
        """Generate a complete reply."""

    @abstractmethod
    def astream(
        self,
        messages: list[dict[str, str]],
        options: GenerationOptions,
        usage: Usage,
    ) -> AsyncIterator[str]:
        """Stream a reply as text chunks, filling `usage` as counts arrive.

        `usage` is an out-parameter rather than a return value because an async
        generator cannot hand one back, and a field on the backend would be a
        race the moment two turns overlap. The caller owns the object it passes.
        """
