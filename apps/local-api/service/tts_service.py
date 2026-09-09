"""Text-to-speech via Piper, behind a small voice registry.

Piper is fast (measured RTF 0.057 on this CPU, far under real time) and has
Vietnamese voices, which is what this app needs -- the alternative Kokoro
backend (no Vietnamese, ~6x heavier) has been removed.

Synthesis renders the **whole** utterance in one call and returns complete WAV
bytes. It used to be driven sentence-by-sentence from the client, which cost a
sentence-boundary bug (the tail of a reply was spoken, then the whole reply was
spoken again) and made prosody impossible: a contour that spans a clause cannot
be planned from one clause at a time. Piper's RTF leaves no latency argument
for splitting it either.

Synthesis is deliberately synchronous here and pushed to a thread by callers.
"""

import io
import logging
import re
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from piper import PiperVoice
from piper.config import SynthesisConfig

from service import prosody
from service.prosody import ProsodyPlan, SpokenSpan

logger = logging.getLogger(__name__)

# Control-rate for the Fujisaki contour. 200 Hz resolves the ~50 ms accent
# rises the model produces without evaluating the filters per audio sample.
_CONTOUR_RATE_HZ = 200.0
_MITSUKA_NAME = re.compile(r"\bMitsuka\b", re.IGNORECASE)
_ASCII_ELLIPSIS = re.compile(r"\.{3,}")
# Preserve decimal points (e.g. 3.14), but remove sentence dots that this
# Piper voice otherwise verbalises as "chấm".
_SPOKEN_DOT = re.compile(r"(?<!\d)\.+|\.+(?!\d)")
_ELLIPSIS_PAUSE_SECONDS = 0.52
_EXCLAMATION_PITCH_RISE = 0.055


def normalize_piper_text(text: str) -> str:
    """Make display text natural for Piper without changing the chat transcript."""
    text = _MITSUKA_NAME.sub("Mít-su-ka", text)
    # Keep ellipses as one semantic mark before stripping ordinary dots. This
    # avoids Piper saying "chấm" while retaining a marker for a longer pause.
    text = _ASCII_ELLIPSIS.sub("…", text)
    text = _SPOKEN_DOT.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def exclamation_pitch_ratio(sample_count: int) -> np.ndarray:
    """Return a gentle, duration-neutral phrase-final lift for `!`."""
    ratio = np.ones(max(sample_count, 0), dtype=np.float64)
    start = int(sample_count * 0.55)
    if start >= sample_count:
        return ratio
    progress = np.linspace(0.0, 1.0, sample_count - start, dtype=np.float64)
    smooth = progress * progress * (3.0 - 2.0 * progress)
    ratio[start:] += _EXCLAMATION_PITCH_RISE * smooth
    return ratio


@dataclass
class _PiperVoiceEntry:
    id: str
    path: Path
    config: Path
    _voice: PiperVoice | None = None

    def load(self) -> PiperVoice:
        if self._voice is None:
            logger.info("piper | loading voice %r from %s", self.id, self.path.name)
            self._voice = PiperVoice.load(str(self.path), config_path=str(self.config))
            logger.info(
                "piper | voice %r ready (sample_rate=%d)", self.id, self._voice.config.sample_rate,
            )
        return self._voice


def _discover_piper_voices(models_dir: Path) -> dict[str, _PiperVoiceEntry]:
    """Pair every `.onnx` under `models_dir` with its `.onnx.json` sidecar.

    Before:
    - "vi_VN-25hours_single-low.onnx"

    After:
    - voice id "25hours_single-low"
    """
    voices: dict[str, _PiperVoiceEntry] = {}
    for onnx_file in sorted(models_dir.glob("*.onnx")):
        config_file = onnx_file.with_suffix(".onnx.json")
        if not config_file.exists():
            continue
        stem = onnx_file.stem
        parts = stem.split("-", 1)
        voice_id = parts[1] if len(parts) > 1 else stem
        voices[voice_id] = _PiperVoiceEntry(id=voice_id, path=onnx_file, config=config_file)
    return voices


def _to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


class TTSService:
    """Resolves a voice id to its Piper voice and renders WAV bytes.

    Use when: the app needs text-to-speech. Construct exactly once, inside the
    service container.

    Expects: `synthesize_wav` to be called off the event loop -- it blocks for
    as long as synthesis plus prosody post-processing takes.

    Returns: complete 16-bit PCM WAV bytes for the whole utterance, ready to
    hand to a client in one response.
    """

    def __init__(self, piper_models_dir: Path) -> None:
        self._piper = _discover_piper_voices(piper_models_dir)

        if not self._piper:
            logger.warning("tts | no Piper voice found in %s", piper_models_dir)
        else:
            logger.info("tts | Piper voices: %s", list(self._piper))

    def list_voices(self) -> list[dict]:
        return [{"id": vid, "name": vid, "engine": "piper"} for vid in self._piper]

    def has_voice(self, voice_id: str) -> bool:
        return voice_id in self._piper

    def preload(self, voice_id: str) -> None:
        """Load the default voice before the first reply needs to speak."""
        entry = self._piper.get(voice_id)
        if entry is None:
            raise KeyError(voice_id)
        entry.load()

    def synthesize_wav(self, voice_id: str, text: str, plan: ProsodyPlan | None = None) -> bytes:
        """Render `text` as one complete WAV, shaped by `plan`.

        `plan=None` renders with the voice's own defaults -- the behaviour
        every caller had before prosody existed.
        """
        text = normalize_piper_text(text)
        entry = self._piper.get(voice_id)
        if entry is None:
            raise KeyError(voice_id)

        voice = entry.load()
        sample_rate = voice.config.sample_rate
        plan = plan or ProsodyPlan()

        audio, spans = self._render_sentences(voice, text, sample_rate, plan)
        if audio.size == 0:
            return _to_wav_bytes(audio, sample_rate)

        audio = self._apply_prosody(audio, sample_rate, spans, plan)
        return _to_wav_bytes(audio, sample_rate)

    def _render_sentences(
        self, voice: PiperVoice, text: str, sample_rate: int, plan: ProsodyPlan,
    ) -> tuple[np.ndarray, list[SpokenSpan]]:
        """Synthesize every sentence and lay them out on one timeline.

        Piper yields one chunk per sentence and no silence between them, so the
        pause is inserted here -- and because the layout is built rather than
        measured afterwards, each sentence's exact start/end is known and the
        Fujisaki commands can be placed against real boundaries instead of an
        even split of the total duration.

        `length_scale` is pre-multiplied by `pitch_scale`: the resampling step
        that raises the pitch shortens the audio by the same factor, so asking
        Piper for a proportionally longer render is what makes the final tempo
        come out at `plan.length_scale` exactly.
        """
        syn_config = SynthesisConfig(
            length_scale=plan.length_scale * plan.pitch_scale,
            noise_w_scale=plan.noise_w_scale,
            normalize_audio=True,
        )

        chunks = [
            chunk.audio_float_array
            for chunk in voice.synthesize(text, syn_config=syn_config)
            if chunk.audio_float_array.size
        ]
        if not chunks:
            return np.zeros(0, dtype=np.float32), []

        # Sentence texts are only needed for syllable counts and question
        # marks. When espeak disagrees with our splitter about how many
        # sentences there are, fall back to the whole text rather than pairing
        # a chunk with the wrong sentence's syllable count.
        sentences = prosody.split_sentences(text)
        if len(sentences) != len(chunks):
            sentences = [text] * len(chunks)

        pieces: list[np.ndarray] = []
        spans: list[SpokenSpan] = []
        cursor = 0

        for index, (chunk, sentence) in enumerate(zip(chunks, sentences)):
            if index:
                previous = sentences[index - 1].rstrip()
                pause_seconds = (
                    max(plan.sentence_pause_s, _ELLIPSIS_PAUSE_SECONDS)
                    if previous.endswith("…")
                    else plan.sentence_pause_s
                )
                pause = np.zeros(int(pause_seconds * sample_rate), dtype=np.float32)
                pieces.append(pause)
                cursor += pause.size
            resampled = prosody.resample_by(np.asarray(chunk, dtype=np.float32), plan.pitch_scale)
            stripped_sentence = sentence.rstrip()
            if stripped_sentence.rstrip("…").rstrip().endswith("!"):
                resampled = prosody.pitch_shift_variable(
                    resampled, exclamation_pitch_ratio(resampled.size), sample_rate,
                )
            pieces.append(resampled)
            spans.append(SpokenSpan(
                start=cursor / sample_rate,
                end=(cursor + resampled.size) / sample_rate,
                syllables=max(prosody.count_syllables(sentence), 1),
                is_question=sentence.rstrip().endswith("?"),
            ))
            cursor += resampled.size

            # A final `...` has no following sentence to receive the inter-
            # sentence pause, so carry the hesitation in the returned WAV.
            if index == len(chunks) - 1 and stripped_sentence.endswith("…"):
                trailing_pause = np.zeros(
                    int(max(plan.sentence_pause_s, _ELLIPSIS_PAUSE_SECONDS) * sample_rate),
                    dtype=np.float32,
                )
                pieces.append(trailing_pause)
                cursor += trailing_pause.size

        return np.concatenate(pieces).astype(np.float32), spans

    def _apply_prosody(
        self, audio: np.ndarray, sample_rate: int, spans: list[SpokenSpan], plan: ProsodyPlan,
    ) -> np.ndarray:
        """Bend the rendered waveform onto the plan's Fujisaki contour."""
        if not plan.alters_waveform:
            return audio

        if plan.contour_depth > 1e-3 and spans:
            phrases, accents = prosody.plan_commands(spans, plan)
            ln_f0 = prosody.fujisaki_ln_f0(
                phrases, accents, audio.size / sample_rate, _CONTOUR_RATE_HZ,
            )
            ratio = prosody.contour_to_ratio(ln_f0, _CONTOUR_RATE_HZ, audio.size, sample_rate)
            logger.debug(
                "tts | fujisaki contour | phrases=%d accents=%d ratio=[%.3f, %.3f]",
                len(phrases), len(accents), float(ratio.min()), float(ratio.max()),
            )
            audio = prosody.pitch_shift_variable(audio, ratio, sample_rate)

        if abs(plan.volume - 1.0) >= 1e-3:
            audio = audio * plan.volume

        return prosody.soft_limit(audio)
