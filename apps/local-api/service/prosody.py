"""Emotion-driven prosody for Piper output, via the Fujisaki F0 command model.

Piper's VITS voices speak with a fixed delivery: the same tempo, the same
pitch and the same declination whatever the assistant just said. This module
turns the agent's own V/A/D reading (`schemas/vad.py`'s `AgentVAD`, in [0, 1])
into the three things a listener actually hears -- tempo, overall pitch, and
the shape of the pitch contour -- and applies them to the rendered waveform.

The contour comes from the Fujisaki model (Fujisaki & Hirose, 1984), which
generates log-F0 as a baseline plus two kinds of command:

    ln F0(t) = ln Fb
             + SUM_i Ap_i * Gp(t - T0_i)                      # phrase commands
             + SUM_j Aa_j * [Ga(t - T1_j) - Ga(t - T2_j)]     # accent commands

    Gp(t) = a^2 * t * exp(-a*t)                     (impulse response, t >= 0)
    Ga(t) = min(1 - (1 + b*t) * exp(-b*t), g)       (step response, t >= 0)

Phrase commands produce the slow declination across a clause; accent commands
produce the local rise on each accent group. Using this rather than a
hand-drawn curve is what keeps the result sounding like breath and glottal
tension instead of like an LFO -- both filters model the physical system, so
their outputs already have the right shape and the right asymmetry (fast rise,
slow fall).

Two parts of the pitch change are applied by completely different means, on
purpose:

- The *constant* part is free and artefact-free: ask Piper for a
  proportionally longer utterance (`length_scale`) and resample it back down.
  Nothing is invented, so nothing can sound synthetic.
- Only the *contour* -- what is left after removing the mean -- goes through
  the granular shifter in `pitch_shift_variable`. Those residual ratios stay
  within a few percent, which is the range where granular shifting is
  transparent.

Why emotion no longer touches either of them:

Both were derived per utterance, which was defensible while an utterance meant
a whole reply. It stopped being defensible when the reply started being spoken
one sentence at a time, because each sentence is now scored on its own words:
adjacent sentences of the same answer came back with different pitch scales and
were audibly a lower voice and a higher voice taking turns. The phrase command
made it worse rather than better -- each sentence got its own declination, so
every one of them started high and ended low.

So baseline pitch is a constant of the voice (`tts_pitch_scale`, tuned once for a bright
delivery) and the contour is off by default (`tts_contour_depth`). Emotion
still moves tempo, pause length and volume, none of which fragment this way.
The Fujisaki machinery below is kept whole and still tested: it is one config
value away, and a whole-reply caller (`/v1/chat`'s `agent_vad`) is exactly the
case it was right for.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import numpy as np
from scipy.signal import correlate, resample_poly

logger = logging.getLogger(__name__)

# Fujisaki filter constants: the physiological ones from the literature
# (a ~ 2-3 /s, b ~ 20 /s, g = 0.9). Emotion moves the command *amplitudes*,
# never these filter shapes, which is what keeps every setting sounding like
# the same speaker rather than like a different voice.
PHRASE_ALPHA = 3.0
ACCENT_BETA = 20.0
ACCENT_CEILING = 0.9

# Vietnamese is syllable-timed and orthographically monosyllabic, so a
# two-syllable accent group matches the language's actual foot better than the
# three or four a stress-timed language would want.
ACCENT_GROUP_SYLLABLES = 2

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+")


@dataclass(frozen=True)
class ProsodyPlan:
    """What one utterance should sound like, derived from V/A/D.

    `length_scale`/`noise_w_scale` are handed to Piper directly; `pitch_scale`,
    `contour_depth` and the Fujisaki amplitudes are applied to the rendered
    audio afterwards.
    """

    length_scale: float = 1.0
    noise_w_scale: float = 0.8
    pitch_scale: float = 1.0
    contour_depth: float = 0.0
    phrase_amplitude: float = 0.0
    accent_amplitude: float = 0.0
    sentence_pause_s: float = 0.18
    volume: float = 1.0

    @property
    def alters_waveform(self) -> bool:
        """False when the post-processing stage provably cannot change the
        audio, so it can be skipped outright instead of run as a no-op."""
        return (
            abs(self.pitch_scale - 1.0) >= 1e-3
            or self.contour_depth > 1e-3
            or abs(self.volume - 1.0) >= 1e-3
        )


def _centred(value: float) -> float:
    """[0, 1] model output -> [-1, 1] with 0.5 as neutral."""
    return float(np.clip(value, 0.0, 1.0)) * 2.0 - 1.0


def derive_prosody(
    valence: float,
    arousal: float,
    dominance: float,
    *,
    depth: float = 1.0,
    pitch_scale: float = 1.0,
    contour_depth: float = 0.0,
) -> ProsodyPlan:
    """Map one V/A/D reading in [0, 1] onto a `ProsodyPlan`.

    Use when: the agent's response text has been scored by `TextVADService`
    and is about to be spoken.

    Expects: the checkpoints' native [0, 1] range (`AgentVAD`), not the signed
    [-1, 1] convention used by the Live2D driver and `/vad`.

    Emotion moves **tempo**, and only tempo. Pitch is whatever `pitch_scale`
    says and is the same for every utterance -- see the module docstring for
    why the emotion-driven version had to go. `contour_depth` above 0 arms the
    Fujisaki commands, which are the other thing that moves pitch; it is off by
    default for the same reason.

    `depth` scales every deviation from neutral, so one setting takes the whole
    effect from off (0.0) to theatrical (~1.5) without changing the relative
    weighting of the three dimensions.

    Before / after, at depth 1.0:
    - (0.5, 0.5, 0.5) neutral -> tempo 1.00
    - (0.8, 0.9, 0.7) excited -> faster, shorter pauses, slightly louder
    - (0.2, 0.2, 0.3) subdued -> slower, longer pauses
    """
    v, a, d = _centred(valence), _centred(arousal), _centred(dominance)
    depth = float(np.clip(depth, 0.0, 2.0))
    contour_depth = float(np.clip(contour_depth, 0.0, 2.0))

    # Arousal is the tempo dimension. Low valence drags on top of it, which is
    # what separates "calm" (slow, level) from "sad" (slow, falling).
    length_scale = 1.0 - depth * (0.20 * a + 0.05 * v)
    # Phoneme-width noise is Piper's only "liveliness" control, but it is not a
    # usable emotion dial: it scales *mean* duration as hard as `length_scale`
    # does. Measured on this voice over the narrow band it was being driven
    # across, at a fixed length_scale of 1.0: noise_w 0.70 -> 2.38s, 0.80 ->
    # 2.78s, 0.92 -> 2.94s, against length_scale's own 0.80 -> 2.57s, 1.175 ->
    # 3.55s. Tying it to arousal therefore cancelled the tempo it was supposed
    # to garnish -- a subdued reply asked for 1.175 (slower) and a noise_w of
    # 0.758, and came out *faster* than neutral. Held constant, so tempo has
    # exactly one owner.
    noise_w_scale = 0.8
    # An activated speaker leaves shorter gaps between sentences.
    sentence_pause_s = 0.20 - depth * 0.09 * a
    volume = 1.0 + depth * 0.10 * a

    # Declination depth: an assertive, activated speaker starts a clause high
    # and falls further across it; a submissive one barely declines at all.
    #
    # These are far below the amplitudes the Fujisaki literature fits to raw
    # speech, because the contour here is added *on top of* the F0 Piper
    # already produces rather than replacing it. Measured on this voice, Piper
    # alone spans ~6.8 semitones between the 10th and 90th F0 percentile; the
    # first tuning pass added another ~7 and the result read as a sing-song.
    #
    # `contour_depth` of 0 zeroes both, which is what makes the contour stage a
    # real bypass rather than a quiet version of itself: `alters_waveform` then
    # skips the granular shifter outright.
    phrase_amplitude = contour_depth * (0.05 + 0.07 * a + 0.04 * d)
    # Local prominence on each accent group.
    accent_amplitude = contour_depth * (0.035 + 0.05 * a + 0.025 * d)

    return ProsodyPlan(
        length_scale=float(np.clip(length_scale, 0.78, 1.30)),
        noise_w_scale=float(np.clip(noise_w_scale, 0.70, 0.92)),
        # Not derived, not clamped against a derived range: this is the voice's
        # fixed brightness, and every utterance in a reply must get the same one.
        pitch_scale=float(np.clip(pitch_scale, 0.82, 1.22)),
        contour_depth=contour_depth,
        phrase_amplitude=float(np.clip(phrase_amplitude, 0.0, 0.22)),
        accent_amplitude=float(np.clip(accent_amplitude, 0.0, 0.16)),
        sentence_pause_s=float(np.clip(sentence_pause_s, 0.06, 0.32)),
        volume=float(np.clip(volume, 0.85, 1.15)),
    )


# -- Fujisaki command generation ---------------------------------------------

@dataclass(frozen=True)
class PhraseCommand:
    """Impulse at `onset`; its response is the clause-long declination."""

    onset: float
    amplitude: float


@dataclass(frozen=True)
class AccentCommand:
    """Step up at `onset`, step back down at `offset`."""

    onset: float
    offset: float
    amplitude: float


@dataclass(frozen=True)
class SpokenSpan:
    """One synthesised sentence, placed on the final utterance timeline."""

    start: float
    end: float
    syllables: int
    is_question: bool = False


def _phrase_response(t: np.ndarray, alpha: float = PHRASE_ALPHA) -> np.ndarray:
    """Gp(t) = a^2 t e^(-a t), zero before onset. Peaks at t = 1/a."""
    active = np.maximum(t, 0.0)
    return np.where(t > 0.0, alpha * alpha * active * np.exp(-alpha * active), 0.0)


def _accent_response(
    t: np.ndarray, beta: float = ACCENT_BETA, gamma: float = ACCENT_CEILING,
) -> np.ndarray:
    """Ga(t) = min(1 - (1 + b t) e^(-b t), g), zero before onset."""
    active = np.maximum(t, 0.0)
    rise = 1.0 - (1.0 + beta * active) * np.exp(-beta * active)
    return np.where(t > 0.0, np.minimum(rise, gamma), 0.0)


def plan_commands(
    spans: list[SpokenSpan], plan: ProsodyPlan,
) -> tuple[list[PhraseCommand], list[AccentCommand]]:
    """Place Fujisaki commands for each spoken sentence.

    One phrase command per sentence, fired `1/alpha` before the sentence's
    first sound so its impulse response peaks exactly at the onset -- that is
    what makes a clause *start* high rather than swell into it.

    Accent commands cover fixed-size syllable groups, interpolated evenly
    across the sentence: Piper's ONNX graph exposes only the waveform (no
    phoneme alignments), and Vietnamese syllables are close enough to
    isochronous that even spacing lands each accent on a real syllable rather
    than between two.

    A question keeps its phrase command but gets a boosted accent command on
    the final group whose step-down falls past the end of the audio, so the
    contour is still rising when the utterance stops.
    """
    phrases: list[PhraseCommand] = []
    accents: list[AccentCommand] = []

    for span in spans:
        duration = span.end - span.start
        if duration <= 0.0 or span.syllables <= 0:
            continue

        phrases.append(
            PhraseCommand(onset=span.start - 1.0 / PHRASE_ALPHA, amplitude=plan.phrase_amplitude)
        )

        per_syllable = duration / span.syllables
        groups = list(range(0, span.syllables, ACCENT_GROUP_SYLLABLES))
        for group_index, first in enumerate(groups):
            last = min(first + ACCENT_GROUP_SYLLABLES, span.syllables)
            onset = span.start + first * per_syllable
            offset = span.start + last * per_syllable
            is_final = group_index == len(groups) - 1

            if is_final and span.is_question:
                accents.append(
                    AccentCommand(onset, span.end + 1.0, plan.accent_amplitude * 1.7)
                )
                continue

            # Later groups in a clause are progressively less prominent.
            # Accent reduction is what stops an utterance sounding recited.
            decay = 1.0 - 0.25 * group_index / max(len(groups) - 1, 1)
            accents.append(AccentCommand(onset, offset, plan.accent_amplitude * decay))

    return phrases, accents


def fujisaki_ln_f0(
    phrases: list[PhraseCommand],
    accents: list[AccentCommand],
    duration_s: float,
    control_rate_hz: float = 200.0,
) -> np.ndarray:
    """Sum the command responses into a log-F0 contour at `control_rate_hz`.

    ln Fb is deliberately left out: the constant term is `pitch_scale`'s job,
    and `contour_to_ratio` subtracts the mean anyway.
    """
    n = max(int(round(duration_s * control_rate_hz)), 1)
    t = np.arange(n, dtype=np.float64) / control_rate_hz
    ln_f0 = np.zeros(n, dtype=np.float64)

    for phrase in phrases:
        ln_f0 += phrase.amplitude * _phrase_response(t - phrase.onset)
    for accent in accents:
        ln_f0 += accent.amplitude * (
            _accent_response(t - accent.onset) - _accent_response(t - accent.offset)
        )

    return ln_f0


def contour_to_ratio(
    ln_f0: np.ndarray, control_rate_hz: float, n_samples: int, sample_rate: int,
) -> np.ndarray:
    """Turn a log-F0 contour into a per-sample pitch ratio, mean-normalised.

    Removing the mean is what makes this composable with `pitch_scale`: the
    ratios returned here average 1.0, so they bend the voice around whatever
    base pitch the clean resampling step already set instead of fighting it.
    """
    if ln_f0.size == 0 or n_samples <= 0:
        return np.ones(max(n_samples, 0), dtype=np.float64)

    centred = ln_f0 - float(np.mean(ln_f0))
    source_t = np.arange(ln_f0.size, dtype=np.float64) / control_rate_hz
    target_t = np.arange(n_samples, dtype=np.float64) / sample_rate
    return np.exp(np.interp(target_t, source_t, centred))


# -- Waveform operations ------------------------------------------------------

def resample_by(audio: np.ndarray, ratio: float, resolution: int = 400) -> np.ndarray:
    """Resample so playback runs `ratio` times faster -- pitch * ratio,
    duration / ratio. Polyphase, so it adds no interpolation noise."""
    if audio.size == 0 or abs(ratio - 1.0) < 1e-4:
        return audio
    down = int(round(resolution * ratio))
    if down <= 0 or down == resolution:
        return audio
    return resample_poly(audio, up=resolution, down=down).astype(np.float32)


def pitch_shift_variable(
    audio: np.ndarray,
    ratio: np.ndarray,
    sample_rate: int,
    frame: int = 256,
    hop: int = 128,
    max_lag_s: float = 0.0087,
) -> np.ndarray:
    """Shift pitch by a per-sample `ratio` while preserving duration exactly.

    Use when: `ratio` stays within roughly +-15% of 1.0. Each output grain is
    read from the source with a time base compressed by the local ratio and
    overlap-added back at its original position, so total length never moves.

    Every grain's read position is first nudged by up to `max_lag_s` to
    whichever offset best correlates with what has already been written (SOLA).
    Without that search adjacent grains land at unrelated points in the glottal
    cycle and the sum partially cancels -- heard as a metallic thinning on
    sustained vowels, the artefact that makes naive granular shifting
    recognisable.

    The defaults were measured rather than guessed, on this app's own voice at
    22.05 kHz. Retained RMS against the unshifted source, at ratio 1.06:

        frame 1024 / hop 256   0.88
        frame  512 / hop 256   0.96
        frame  256 / hop 128   0.985

    Short grains win because the resampled time base drifts *within* a grain,
    and no single alignment offset can correct a drift that grows across the
    window: at ratio 1.06 a 1024-sample grain slides through half a pitch
    period from its start to its end. Two further consequences of the same
    reasoning are baked in below -- the grid is centred on the grain midpoint
    so the drift is symmetric about the window's peak instead of accumulating
    from its leading edge, and hop is exactly half the frame so the Hann
    windows sum to unity and the normalisation cannot modulate amplitude.

    Measured periodicity (mean autocorrelation peak over voiced frames) comes
    out slightly *above* the source's, so the alignment adds no roughness.
    """
    n = audio.size
    if n == 0:
        return audio

    ratio = np.atleast_1d(np.asarray(ratio, dtype=np.float64))
    if ratio.size == 1:
        ratio = np.full(n, float(ratio[0]))
    elif ratio.size != n:
        ratio = np.interp(
            np.linspace(0.0, 1.0, n), np.linspace(0.0, 1.0, ratio.size), ratio,
        )

    max_lag = max(int(round(max_lag_s * sample_rate)), 1)
    overlap = frame - hop
    # Head/tail room for the SOLA search and for the furthest read a grain can
    # make once its time base is stretched by the largest ratio in the curve.
    pad_left = max_lag + frame
    pad_right = max_lag + 2 * frame + int(frame * float(np.max(ratio)) + 1)
    padded = np.concatenate([
        np.zeros(pad_left, dtype=np.float64),
        audio.astype(np.float64),
        np.zeros(pad_right, dtype=np.float64),
    ])
    source_index = np.arange(padded.size, dtype=np.float64)

    window = np.hanning(frame + 1)[:frame]
    # Read offsets measured from the grain's midpoint, so a ratio away from 1.0
    # pulls the two halves symmetrically apart instead of dragging the whole
    # grain later and later relative to its window.
    centre = frame / 2.0
    grid = np.arange(frame, dtype=np.float64) - centre
    ones_overlap = np.ones(overlap, dtype=np.float64)

    out = np.zeros(n + frame, dtype=np.float64)
    norm = np.zeros(n + frame, dtype=np.float64)

    for start in range(0, n, hop):
        local_ratio = float(ratio[min(start, n - 1)])

        delta = 0
        written = norm[start : start + overlap]
        if overlap > 0 and written.size == overlap and np.any(written > 1e-6):
            reference = out[start : start + overlap] / np.maximum(written, 1e-9)
            search = padded[pad_left + start - max_lag : pad_left + start + overlap + max_lag]
            if search.size == overlap + 2 * max_lag and np.any(reference):
                scores = correlate(search, reference, mode="valid", method="fft")
                # Normalise by each candidate's own energy, or the search just
                # walks to whichever offset happens to be loudest.
                energy = np.sqrt(np.maximum(
                    correlate(search * search, ones_overlap, mode="valid"), 1e-9,
                ))
                delta = int(np.argmax(scores / energy)) - max_lag

        read_at = pad_left + start + delta + centre + grid * local_ratio
        grain = np.interp(read_at, source_index, padded)

        out[start : start + frame] += grain * window
        norm[start : start + frame] += window

    return (out[:n] / np.maximum(norm[:n], 1e-6)).astype(np.float32)


def soft_limit(audio: np.ndarray, ceiling: float = 0.97) -> np.ndarray:
    """Tanh-knee limiter. Volume and the contour can each push peaks past full
    scale, and hard clipping there is audible as a click on exactly the loud,
    high-arousal deliveries this module exists to improve."""
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak <= ceiling:
        return audio
    return (np.tanh(audio / peak) * (ceiling / np.tanh(1.0))).astype(np.float32)


# -- Text -> timeline ---------------------------------------------------------

def split_sentences(text: str) -> list[str]:
    """Split on sentence terminators -- the same boundary espeak uses to decide
    where one Piper audio chunk ends and the next begins."""
    return [part for part in (p.strip() for p in _SENTENCE_SPLIT.split(text.strip())) if part]


def count_syllables(text: str) -> int:
    """Whitespace-separated tokens carrying at least one letter or digit.

    Vietnamese orthography writes one syllable per token, so this is exact for
    Vietnamese and a usable approximation for the Latin-script fragments
    (names, numbers) that turn up inside it.
    """
    return sum(1 for token in text.split() if any(ch.isalnum() for ch in token))
