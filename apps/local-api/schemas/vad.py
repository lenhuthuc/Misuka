from typing import Literal

from pydantic import BaseModel


class VADRequest(BaseModel):
    text: str


class VADResponse(BaseModel):
    v: float
    a: float
    d: float


class VADScores(BaseModel):
    """VAD dimensions in [-1, 1]."""
    valence: float
    arousal: float
    dominance: float


class ASRInfo(BaseModel):
    engine: Literal["sherpa-onnx"] = "sherpa-onnx"
    language: Literal["vi"] = "vi"


class UserVAD(BaseModel):
    """VAD dimensions in [0, 1] — the trained checkpoints' native range.

    `mode="multimodal"` when both audio and its transcript were available;
    `mode="text"` when the input was text-only and there was no audio to run
    the multimodal (WavLM+PhoBERT) model on.
    """
    mode: Literal["multimodal", "text"]
    valence: float
    arousal: float
    dominance: float


class AgentVAD(BaseModel):
    """VAD dimensions in [0, 1] for the agent's complete response text —
    always text-only, never derived from user audio/transcript/user_vad."""
    mode: Literal["text"] = "text"
    valence: float
    arousal: float
    dominance: float


class EmotionVADResponse(BaseModel):
    transcript: str
    # None when `user_vad.mode == "text"` — no audio was uploaded, so ASR
    # never ran (`transcript` is simply the text the caller sent).
    asr: ASRInfo | None
    user_vad: UserVAD
