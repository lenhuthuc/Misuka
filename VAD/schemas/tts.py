from typing import Literal

from pydantic import BaseModel


class TTSRequest(BaseModel):
    model: str = "piper"
    input: str
    voice: str = "default"
    response_format: Literal["wav", "mp3", "opus", "aac", "flac"] = "wav"
    speed: float = 1.0
