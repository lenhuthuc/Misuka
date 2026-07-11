"""Deterministic V/A/D → emotion label mapping.

Prototype coordinates follow the PAD (pleasure-arousal-dominance) literature
(Mehrabian 1996 and derivatives), expressed in [-1, 1] — the same range the
VAD model outputs (tanh head). Mapping is nearest-prototype by Euclidean
distance, so identical inputs always yield the same label.
"""
from __future__ import annotations

import math

# (valence, arousal, dominance) prototypes. Order matters: on an exact
# distance tie the earlier entry wins, keeping the mapping deterministic.
EMOTION_PROTOTYPES: dict[str, tuple[float, float, float]] = {
    # ── positive ──────────────────────────────────────────────────────────
    "joy":            ( 0.76,  0.48,  0.35),
    "excitement":     ( 0.62,  0.75,  0.38),
    "contentment":    ( 0.78, -0.15,  0.32),
    "serenity":       ( 0.60, -0.50,  0.25),
    "affection":      ( 0.85,  0.25,  0.15),
    "gratitude":      ( 0.69,  0.15, -0.09),
    "pride":          ( 0.72,  0.38,  0.60),
    "hope":           ( 0.51,  0.23,  0.14),
    "amusement":      ( 0.70,  0.44,  0.24),
    "relief":         ( 0.55, -0.25,  0.12),
    "curiosity":      ( 0.40,  0.35,  0.25),
    "surprise":       ( 0.35,  0.70, -0.13),
    # ── negative, high arousal ────────────────────────────────────────────
    "anger":          (-0.51,  0.59,  0.25),
    "frustration":    (-0.47,  0.42,  0.05),
    "fear":           (-0.64,  0.60, -0.43),
    "anxiety":        (-0.45,  0.40, -0.30),
    "disgust":        (-0.60,  0.35,  0.11),
    "contempt":       (-0.55,  0.18,  0.35),
    "envy":           (-0.40,  0.30,  0.06),
    # ── negative, low arousal ─────────────────────────────────────────────
    "sadness":        (-0.63, -0.27, -0.33),
    "grief":          (-0.75, -0.10, -0.45),
    "disappointment": (-0.53, -0.20, -0.25),
    "boredom":        (-0.35, -0.55, -0.20),
    "shame":          (-0.57,  0.01, -0.34),
    "guilt":          (-0.55,  0.20, -0.40),
    "loneliness":     (-0.58, -0.35, -0.42),
    # ── baseline ──────────────────────────────────────────────────────────
    "neutral":        ( 0.00,  0.00,  0.00),
}


def map_vad_to_emotion(valence: float, arousal: float, dominance: float) -> str:
    """Return the emotion label whose prototype is nearest in VAD space."""
    best_label, best_dist = "neutral", math.inf
    for label, (pv, pa, pd) in EMOTION_PROTOTYPES.items():
        dist = (valence - pv) ** 2 + (arousal - pa) ** 2 + (dominance - pd) ** 2
        if dist < best_dist:
            best_label, best_dist = label, dist
    return best_label
