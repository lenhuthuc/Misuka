"""Emotion state inference for system responses.

Bridges the PhoBERT VAD model (text → V/A/D) with the deterministic emotion
mapper, and blends the current response's V/A/D with the V/A/D of retrieved
memories into the system's current emotional state.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from brain.emotion_mapper import map_vad_to_emotion

if TYPE_CHECKING:
    from service.vad_service import VADService
    from brain.state import RetrievedDoc

logger = logging.getLogger(__name__)

_VAD_KEYS = ("valence", "arousal", "dominance")


@dataclass(frozen=True)
class EmotionReading:
    valence: float
    arousal: float
    dominance: float
    emotion: str

    @classmethod
    def from_vad(cls, valence: float, arousal: float, dominance: float) -> "EmotionReading":
        return cls(valence, arousal, dominance, map_vad_to_emotion(valence, arousal, dominance))

    @property
    def vad(self) -> tuple[float, float, float]:
        return self.valence, self.arousal, self.dominance

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def extract_memory_vads(docs: list["RetrievedDoc"]) -> list[tuple[float, float, float]]:
    """Pull V/A/D triples out of retrieved-doc metadata, skipping docs without them."""
    vads = []
    for doc in docs:
        meta = doc.get("metadata") or {}
        if all(isinstance(meta.get(k), (int, float)) for k in _VAD_KEYS):
            vads.append(tuple(float(meta[k]) for k in _VAD_KEYS))
    return vads


class EmotionService:
    """Infer V/A/D + emotion for text and compute the blended system state."""

    def __init__(self, vad_service: "VADService", current_weight: float = 0.6) -> None:
        self._vad = vad_service
        self._current_weight = current_weight

    async def infer(self, text: str) -> EmotionReading:
        """V/A/D + emotion label for a piece of text (model runs off the event loop)."""
        v, a, d = await asyncio.to_thread(self._vad.predict, text)
        reading = EmotionReading.from_vad(v, a, d)
        logger.debug("emotion | v=%.2f a=%.2f d=%.2f → %s", v, a, d, reading.emotion)
        return reading

    def current_state(
        self,
        current: EmotionReading,
        memory_vads: list[tuple[float, float, float]],
    ) -> EmotionReading:
        """Blend the current response V/A/D with retrieved memories' V/A/D.

        current × w + mean(memories) × (1 - w); no memories → current as-is.
        """
        if not memory_vads:
            return current
        n = len(memory_vads)
        mem_mean = tuple(sum(vad[i] for vad in memory_vads) / n for i in range(3))
        w = self._current_weight
        blended = tuple(c * w + m * (1.0 - w) for c, m in zip(current.vad, mem_mean))
        return EmotionReading.from_vad(*blended)
