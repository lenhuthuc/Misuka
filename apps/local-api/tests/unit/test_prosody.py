"""Tests for the V/A/D -> Fujisaki prosody stage.

All pure: no Piper, no ONNX, no audio device. The waveform tests run on
synthetic tones so the expected result is analytic rather than perceptual.
"""
from __future__ import annotations

import numpy as np
import pytest

from service.prosody import (
    PHRASE_ALPHA,
    ProsodyPlan,
    SpokenSpan,
    _accent_response,
    _phrase_response,
    contour_to_ratio,
    count_syllables,
    derive_prosody,
    fujisaki_ln_f0,
    pitch_shift_variable,
    plan_commands,
    resample_by,
    soft_limit,
    split_sentences,
)

SR = 22050


def _tone(freq: float, seconds: float = 0.5, sr: int = SR) -> np.ndarray:
    t = np.arange(int(seconds * sr), dtype=np.float64) / sr
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _measured_freq(x: np.ndarray, sr: int = SR) -> float:
    """Frequency from the mean zero-crossing rate — exact for a clean tone."""
    crossings = np.count_nonzero(np.diff(np.signbit(x)))
    return crossings * sr / (2 * len(x))


# -- derive_prosody -----------------------------------------------------------

def test_neutral_vad_leaves_tempo_and_pitch_untouched():
    """0.5 on every axis is the model's midpoint, so it must mean "speak
    normally" rather than "speak slightly oddly"."""
    plan = derive_prosody(0.5, 0.5, 0.5)

    assert plan.length_scale == pytest.approx(1.0)
    assert plan.pitch_scale == pytest.approx(1.0)


def test_depth_zero_disables_the_whole_stage():
    """The `tts_prosody_depth` escape hatch has to be a real bypass, not a
    quieter version of the effect — otherwise there is no way to A/B it."""
    plan = derive_prosody(0.9, 0.9, 0.9, depth=0.0)

    assert plan.length_scale == pytest.approx(1.0)
    assert plan.pitch_scale == pytest.approx(1.0)
    assert plan.volume == pytest.approx(1.0)
    assert not plan.alters_waveform


def test_high_arousal_speaks_faster():
    excited = derive_prosody(0.8, 0.9, 0.7)

    assert excited.length_scale < 1.0  # Piper's length_scale is inverse to speed


def test_low_arousal_speaks_slower():
    subdued = derive_prosody(0.2, 0.2, 0.3)

    assert subdued.length_scale > 1.0


def test_emotion_never_moves_the_pitch():
    """ROOT CAUSE:

    `pitch_scale` used to be `exp(depth * (0.13*a + 0.05*v))`, derived per
    utterance. That was fine while an utterance was a whole reply. Once the
    reply was spoken sentence by sentence, each sentence was scored on its own
    words and came back with its own pitch — one answer was audibly a low voice
    and a high voice taking turns.

    We fixed this by making pitch a constant of the voice, passed in rather than
    derived, so no reading can move it.
    """
    readings = [(0.5, 0.5, 0.5), (0.9, 0.95, 0.9), (0.05, 0.1, 0.2), (0.9, 0.1, 0.5)]

    pitches = [derive_prosody(*r, pitch_scale=1.06).pitch_scale for r in readings]

    assert pitches == [pytest.approx(1.06)] * len(readings)


def test_the_configured_pitch_is_what_comes_out():
    """@example: `tts_pitch_scale=1.06` -> the plan asks for 1.06, so a reply is
    rendered a touch bright rather than at whatever the last sentence's emotion
    implied."""
    assert derive_prosody(0.5, 0.5, 0.5, pitch_scale=1.06).pitch_scale == pytest.approx(1.06)
    assert derive_prosody(0.5, 0.5, 0.5).pitch_scale == pytest.approx(1.0)


def test_anger_and_delight_are_separated_by_tempo():
    """@example: both are high-arousal, so both are fast — valence is what still
    separates them, now through tempo alone. They are spoken at one pitch,
    which is the point: an answer must not change voice halfway through."""
    delighted = derive_prosody(0.9, 0.85, 0.6, pitch_scale=1.06)
    angry = derive_prosody(0.1, 0.85, 0.85, pitch_scale=1.06)

    assert delighted.length_scale < 1.0
    assert angry.length_scale < 1.0
    assert delighted.length_scale < angry.length_scale
    assert delighted.pitch_scale == pytest.approx(angry.pitch_scale)


def test_tempo_has_exactly_one_owner():
    """ROOT CAUSE:

    `noise_w_scale` was driven by arousal as a "liveliness" garnish, on the
    assumption that its effect on duration was small enough to ignore inside a
    narrow band. Measured on this voice at a fixed length_scale of 1.0 it is
    not: noise_w 0.70 -> 2.38s, 0.92 -> 2.94s, against length_scale's own
    0.80 -> 2.57s, 1.175 -> 3.55s. The two moved in opposite directions for a
    subdued reading, which asked for 1.175 (slower) and came out *faster* than
    neutral — the one dimension emotion is still allowed to move was broken.

    We fixed this by holding noise_w constant, so `length_scale` alone decides
    tempo. Verified against real synthesis (5 renders each, mean duration):
    excited 2.35s, neutral 2.54s, subdued 3.03s.
    """
    excited = derive_prosody(0.9, 0.9, 0.8)
    neutral = derive_prosody(0.5, 0.5, 0.5)
    subdued = derive_prosody(0.15, 0.15, 0.3)

    assert excited.length_scale < neutral.length_scale < subdued.length_scale
    assert excited.noise_w_scale == neutral.noise_w_scale == subdued.noise_w_scale


def test_the_contour_is_off_unless_asked_for():
    """@example: the default plan draws no F0 contour -> `pitch_shift_variable`
    is skipped outright, since each sentence restarting its own declination is
    the other half of what made a reply sound like two different voices."""
    plan = derive_prosody(0.8, 0.9, 0.7)

    assert plan.contour_depth == 0.0
    assert plan.phrase_amplitude == 0.0
    assert plan.accent_amplitude == 0.0


def test_dominance_deepens_declination_without_changing_tempo():
    """The contour still responds to the reading when it is armed — a
    whole-reply caller can turn it back on with `tts_contour_depth`."""
    submissive = derive_prosody(0.5, 0.5, 0.1, contour_depth=1.0)
    assertive = derive_prosody(0.5, 0.5, 0.9, contour_depth=1.0)

    assert assertive.phrase_amplitude > submissive.phrase_amplitude
    assert assertive.length_scale == pytest.approx(submissive.length_scale)


def test_every_derived_plan_stays_inside_the_safe_ranges():
    """The granular shifter is only transparent near ratio 1.0, and Piper
    degrades at extreme length scales. A saturated V/A/D reading must not be
    able to walk outside either."""
    for corner in [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 1.0)]:
        plan = derive_prosody(*corner, depth=2.0)
        assert 0.78 <= plan.length_scale <= 1.30
        assert 0.82 <= plan.pitch_scale <= 1.22
        assert 0.85 <= plan.volume <= 1.15


def test_out_of_range_scores_are_clamped_not_extrapolated():
    """A caller that mistakenly sends the signed [-1, 1] convention gets the
    floor of the [0, 1] range, not a wildly out-of-band plan."""
    assert derive_prosody(-1.0, -1.0, -1.0) == derive_prosody(0.0, 0.0, 0.0)


# -- Fujisaki filters ---------------------------------------------------------

def test_phrase_response_is_silent_before_its_onset():
    t = np.linspace(-1.0, -0.01, 50)
    assert np.all(_phrase_response(t) == 0.0)


def test_phrase_response_peaks_one_over_alpha_after_onset():
    """That peak time is why `plan_commands` fires the command 1/alpha early:
    it is what makes a clause start high instead of swelling into it."""
    t = np.linspace(0.0, 2.0, 4000)
    peak_at = t[int(np.argmax(_phrase_response(t)))]

    assert peak_at == pytest.approx(1.0 / PHRASE_ALPHA, abs=0.01)


def test_accent_response_rises_monotonically_to_its_ceiling():
    t = np.linspace(0.0, 1.0, 2000)
    response = _accent_response(t)

    assert np.all(np.diff(response) >= -1e-12)
    assert response[-1] == pytest.approx(0.9)


def test_accent_response_is_silent_before_its_onset():
    assert np.all(_accent_response(np.linspace(-1.0, -0.01, 50)) == 0.0)


# -- command placement --------------------------------------------------------

def test_each_sentence_gets_one_phrase_command_fired_before_it_starts():
    plan = derive_prosody(0.5, 0.5, 0.5, contour_depth=1.0)
    spans = [SpokenSpan(0.0, 2.0, syllables=6), SpokenSpan(2.2, 4.0, syllables=5)]

    phrases, _ = plan_commands(spans, plan)

    assert len(phrases) == 2
    assert phrases[0].onset < spans[0].start
    assert phrases[1].onset < spans[1].start


def test_a_question_keeps_rising_past_the_end_of_the_audio():
    """ROOT CAUSE this encodes: a step-down inside the utterance resolves the
    contour, and a resolved contour reads as a statement however the sentence
    is punctuated."""
    plan = derive_prosody(0.5, 0.5, 0.5, contour_depth=1.0)
    span = SpokenSpan(0.0, 2.0, syllables=6, is_question=True)

    _, accents = plan_commands([span], plan)

    assert accents[-1].offset > span.end
    # ...and it is the most prominent accent in the sentence.
    assert accents[-1].amplitude == max(a.amplitude for a in accents)


def test_a_statement_resolves_inside_its_own_span():
    plan = derive_prosody(0.5, 0.5, 0.5, contour_depth=1.0)
    span = SpokenSpan(0.0, 2.0, syllables=6, is_question=False)

    _, accents = plan_commands([span], plan)

    assert all(accent.offset <= span.end for accent in accents)


def test_accents_lose_prominence_across_a_clause():
    """Accent reduction. Equal-amplitude accents are what make a synthesised
    sentence sound recited rather than spoken."""
    plan = derive_prosody(0.5, 0.5, 0.5, contour_depth=1.0)

    _, accents = plan_commands([SpokenSpan(0.0, 4.0, syllables=10)], plan)

    assert accents[0].amplitude > accents[-1].amplitude


def test_a_span_with_no_syllables_produces_no_commands():
    plan = derive_prosody(0.5, 0.5, 0.5, contour_depth=1.0)
    assert plan_commands([SpokenSpan(0.0, 1.0, syllables=0)], plan) == ([], [])


# -- contour ------------------------------------------------------------------

def test_contour_ratios_average_to_unity():
    """The contour must be a pure modulation: its mean pitch shift is
    `pitch_scale`'s job, applied by clean resampling instead. A contour with a
    non-zero mean would fight it."""
    plan = derive_prosody(0.8, 0.9, 0.7, contour_depth=1.0)
    phrases, accents = plan_commands([SpokenSpan(0.0, 3.0, syllables=9)], plan)
    ln_f0 = fujisaki_ln_f0(phrases, accents, duration_s=3.0)

    ratio = contour_to_ratio(ln_f0, 200.0, n_samples=3 * SR, sample_rate=SR)

    assert float(np.mean(np.log(ratio))) == pytest.approx(0.0, abs=1e-3)


def test_contour_actually_moves_within_an_utterance():
    plan = derive_prosody(0.8, 0.9, 0.7, contour_depth=1.0)
    phrases, accents = plan_commands([SpokenSpan(0.0, 3.0, syllables=9)], plan)
    ln_f0 = fujisaki_ln_f0(phrases, accents, duration_s=3.0)

    ratio = contour_to_ratio(ln_f0, 200.0, n_samples=3 * SR, sample_rate=SR)

    assert float(np.max(ratio) / np.min(ratio)) > 1.02


def test_an_empty_contour_degrades_to_no_modulation():
    ratio = contour_to_ratio(np.zeros(0), 200.0, n_samples=100, sample_rate=SR)
    assert np.allclose(ratio, 1.0)


# -- waveform operations ------------------------------------------------------

def test_resample_by_shifts_frequency_and_shortens_by_the_same_factor():
    tone = _tone(200.0, seconds=1.0)

    faster = resample_by(tone, 1.1)

    assert _measured_freq(faster) == pytest.approx(220.0, rel=0.02)
    assert faster.size == pytest.approx(tone.size / 1.1, rel=0.01)


def test_pitch_shift_preserves_duration_exactly():
    """This is the whole reason the variable shifter exists rather than a
    second resample: the utterance's timeline is already committed by the time
    the contour is applied."""
    tone = _tone(200.0)

    shifted = pitch_shift_variable(tone, np.full(tone.size, 1.08), SR)

    assert shifted.size == tone.size


def test_pitch_shift_at_ratio_one_is_the_identity():
    tone = _tone(200.0)

    shifted = pitch_shift_variable(tone, np.ones(tone.size), SR)

    assert np.allclose(shifted, tone, atol=1e-3)


@pytest.mark.parametrize("ratio", [0.92, 1.06, 1.12])
def test_pitch_shift_moves_the_measured_frequency_by_the_requested_ratio(ratio):
    tone = _tone(200.0)

    shifted = pitch_shift_variable(tone, np.full(tone.size, ratio), SR)

    assert _measured_freq(shifted) == pytest.approx(200.0 * ratio, rel=0.06)


def test_pitch_shift_keeps_the_signals_energy():
    """ROOT CAUSE: with a 1024-sample grain, grains drift out of phase within
    the window and the overlap-add cancels — measured at 12% RMS loss at ratio
    1.06, heard as the voice thinning out. The short, centred, half-overlapped
    grain the defaults now use retains ~98%."""
    tone = _tone(200.0)

    shifted = pitch_shift_variable(tone, np.full(tone.size, 1.06), SR)

    source_rms = float(np.sqrt(np.mean(tone.astype(np.float64) ** 2)))
    shifted_rms = float(np.sqrt(np.mean(shifted.astype(np.float64) ** 2)))
    assert shifted_rms / source_rms > 0.9


def test_pitch_shift_accepts_a_curve_shorter_than_the_audio():
    """The contour is generated at 200 Hz control rate, not per sample."""
    tone = _tone(200.0)

    shifted = pitch_shift_variable(tone, np.linspace(0.98, 1.02, 100), SR)

    assert shifted.size == tone.size
    assert np.all(np.isfinite(shifted))


def test_pitch_shift_handles_empty_audio():
    empty = np.zeros(0, dtype=np.float32)
    assert pitch_shift_variable(empty, np.ones(1), SR).size == 0


def test_soft_limit_leaves_audio_that_already_fits_alone():
    quiet = _tone(200.0) * 0.5
    assert np.array_equal(soft_limit(quiet), quiet)


def test_soft_limit_pulls_an_over_range_signal_under_full_scale():
    loud = _tone(200.0) * 4.0

    limited = soft_limit(loud)

    assert float(np.max(np.abs(limited))) == pytest.approx(0.97, abs=1e-4)


# -- text -> timeline ---------------------------------------------------------

def test_split_sentences_matches_the_terminators_espeak_chunks_on():
    assert split_sentences("Xin chào. Mình khỏe! Bạn thì sao?") == [
        "Xin chào.", "Mình khỏe!", "Bạn thì sao?",
    ]


def test_split_sentences_keeps_an_unterminated_tail():
    assert split_sentences("Xin chào. Mình khỏe") == ["Xin chào.", "Mình khỏe"]


def test_count_syllables_counts_vietnamese_words():
    """Vietnamese writes one syllable per whitespace-separated token, which is
    what makes even spacing a usable stand-in for real phoneme alignments."""
    assert count_syllables("một cộng một bằng hai") == 5


def test_count_syllables_ignores_bare_punctuation():
    assert count_syllables("xin chào , bạn !") == 3


def test_a_default_plan_reports_that_it_changes_nothing():
    assert not ProsodyPlan().alters_waveform
