"""Discriminated SSE envelope for /v1/chat/stream.

Every event carries `type` and `turn_id`, so the client can tell events from
different turns apart (relevant once request de-duplication / barge-in
across turns is handled — see REFACTOR_PLAN.md Phase 4) and route on `type`
instead of guessing from which fields happen to be present.
"""
from typing import Literal

from pydantic import BaseModel

from schemas.vad import VADScores


class ChatStreamDeltaEvent(BaseModel):
    type: Literal["delta"] = "delta"
    turn_id: str
    content: str


class ChatStreamEmotionEvent(BaseModel):
    type: Literal["emotion"] = "emotion"
    turn_id: str
    emotion: str
    state: VADScores


class ChatStreamErrorDetail(BaseModel):
    code: str
    message: str
    retryable: bool = True


class ChatStreamErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    turn_id: str
    error: ChatStreamErrorDetail


class ChatStreamDoneEvent(BaseModel):
    type: Literal["done"] = "done"
    turn_id: str


ChatStreamEvent = ChatStreamDeltaEvent | ChatStreamEmotionEvent | ChatStreamErrorEvent | ChatStreamDoneEvent
