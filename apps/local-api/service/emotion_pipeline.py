"""Orchestrates the user-audio side of the VAD architecture:

    audio (upload)
      -> mono/16kHz/float32                          (service/audio_preprocessing.py)
      -> Sherpa-ONNX ASR on the FULL audio            (service/sherpa_asr_service.py)
      -> center-cropped audio + that transcript
      -> WavLM+PhoBERT multimodal VAD                 (service/multimodal_vad_service.py)
      -> user_vad

The agent-response side (`analyze_agent_response`) is intentionally just a
thin call into `TextVADService` — response generation itself stays owned by
`api/chat.py`/`brain/llm_service.py`, so this pipeline is not coupled to the LLM.

Every method here is synchronous/CPU-bound (ASR decode + two model forward
passes); callers run it off the event loop via `container.emotion_executor`,
the same pattern the old Whisper-based `/emotion-vad` route used.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from service.audio_preprocessing import center_crop_or_pad, prepare_mono_16k
from service.multimodal_vad_service import MultimodalVADService
from service.sherpa_asr_service import SherpaASRService
from service.text_vad_service import TextVADService

logger = logging.getLogger(__name__)

# 4s @ 16kHz — the multimodal checkpoint's trained/validated audio window.
MULTIMODAL_CROP_SAMPLES = 64000


@dataclass(frozen=True)
class UserAudioResult:
    transcript: str
    valence: float
    arousal: float
    dominance: float


class EmotionPipeline:
    def __init__(
        self,
        asr: SherpaASRService,
        multimodal_vad: MultimodalVADService,
        text_vad: TextVADService,
    ) -> None:
        self._asr = asr
        self._multimodal_vad = multimodal_vad
        self._text_vad = text_vad

    def analyze_user_audio(self, raw_bytes: bytes) -> UserAudioResult:
        """Full pipeline for one uploaded audio clip: ASR transcript +
        multimodal (audio+text) V/A/D, both derived from the same recording."""
        audio = prepare_mono_16k(raw_bytes)

        transcript = self._asr.transcribe(audio)
        if not transcript:
            logger.warning("emotion_pipeline | ASR produced an empty transcript")

        cropped, mask = center_crop_or_pad(audio, MULTIMODAL_CROP_SAMPLES)
        v, a, d = self._multimodal_vad.predict(cropped, mask, transcript)

        return UserAudioResult(transcript=transcript, valence=v, arousal=a, dominance=d)

    def analyze_agent_response(self, text: str) -> tuple[float, float, float]:
        """Text-only V/A/D for the agent's complete response. Never touches
        user audio, the user transcript, user_vad, or WavLM embeddings."""
        return self._text_vad.predict_raw(text)

    def analyze_user_text(self, text: str) -> UserAudioResult:
        """Fallback for a text-only user turn (no audio to run the multimodal
        model on): text-only V/A/D, same model/range as `analyze_agent_response`.
        Never fabricates audio/zero tensors to force the multimodal model."""
        v, a, d = self._text_vad.predict_raw(text)
        return UserAudioResult(transcript=text, valence=v, arousal=a, dominance=d)
