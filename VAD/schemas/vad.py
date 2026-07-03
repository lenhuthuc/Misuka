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


class EmotionVADResponse(BaseModel):
    transcript: str
    audio: VADScores
    text: VADScores
    fused: VADScores
