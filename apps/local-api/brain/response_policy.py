"""Deterministic, zero-I/O control policy derived from a user's VAD state.

The policy deliberately carries *behavioural* instructions rather than raw V/A/D
coordinates.  It is computed locally before generation, so it adds no model
call or delay before the first streamed token.
"""
from __future__ import annotations

from dataclasses import dataclass

from schemas.vad import VADScores


@dataclass(frozen=True)
class ResponsePolicy:
    """Small set of generation controls for one response."""

    instruction: str = ""
    max_tokens: int | None = None
    temperature: float | None = None
    stream_pace: str = "immediate"

    @property
    def is_active(self) -> bool:
        return bool(self.instruction or self.max_tokens is not None or self.temperature is not None)

    def options(self, default_temperature: float, default_max_tokens: int) -> dict[str, float | int]:
        """Return the Ollama options for this turn, preserving service defaults."""
        return {
            "temperature": self.temperature if self.temperature is not None else default_temperature,
            "num_predict": self.max_tokens if self.max_tokens is not None else default_max_tokens,
        }


def derive_response_policy(vad: VADScores | None, default_max_tokens: int) -> ResponsePolicy:
    """Map VAD to response behaviour without using another inference step.

    High arousal is the only case that caps output length: that reduces both
    generation time and cognitive load when a user is activated.  Streaming is
    always immediate; the API never intentionally delays chunks.
    """
    if vad is None:
        return ResponsePolicy()

    directives: list[str] = []
    max_tokens: int | None = None
    temperature: float | None = None

    if vad.arousal >= 0.45:
        directives.extend(("brief", "calm", "grounding", "at most one question"))
        max_tokens = min(default_max_tokens, 256)
        temperature = 0.55
    elif vad.arousal <= -0.45:
        directives.append("calm, unhurried")

    if vad.valence <= -0.35:
        directives.append("acknowledge feelings without cheerleading")
    elif vad.valence >= 0.45:
        directives.append("warm and engaged")

    if vad.dominance <= -0.35:
        directives.append("offer one concrete next step")
    elif vad.dominance >= 0.35:
        directives.append("direct and collaborative")

    # A neutral reading still explicitly asks for a natural conversational
    # response, while avoiding a raw coordinate dump in the prompt.
    if not directives:
        directives.append("natural and conversational")

    return ResponsePolicy(
        instruction="Response style: " + "; ".join(directives) + ".",
        max_tokens=max_tokens,
        temperature=temperature,
    )
