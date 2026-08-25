from typing import Literal

from pydantic import BaseModel, Field


class TTSRequest(BaseModel):
    model: str = "piper"
    input: str
    voice: str = "default"
    response_format: Literal["wav", "mp3", "opus", "aac", "flac"] = "wav"
    speed: float = 1.0

    # The agent's own V/A/D for this reply, in the checkpoints' native [0, 1]
    # range -- exactly what `/v1/chat`'s `agent_vad` and the stream's `done`
    # event carry. Supplying it drives tempo, pitch and the Fujisaki contour
    # (see service/prosody.py); omitting it renders with the voice's defaults,
    # which is what every pre-prosody caller gets.
    valence: float | None = Field(default=None, ge=0.0, le=1.0)
    arousal: float | None = Field(default=None, ge=0.0, le=1.0)
    dominance: float | None = Field(default=None, ge=0.0, le=1.0)

    # Ask the server to read this utterance's own V/A/D off its text when the
    # caller has none to send. That is the position a client speaking a reply
    # sentence by sentence is in: the reply's own reading only exists once the
    # whole reply has been generated, which is long after its first sentence
    # needed to be spoken. Ignored when an explicit reading is supplied -- a
    # caller that measured the emotion outranks one inferred from the words.
    auto_prosody: bool = False

    @property
    def agent_vad(self) -> tuple[float, float, float] | None:
        """The three scores, or None unless all three were sent -- a partial
        reading would silently be treated as neutral on the missing axis."""
        if self.valence is None or self.arousal is None or self.dominance is None:
            return None
        return self.valence, self.arousal, self.dominance
