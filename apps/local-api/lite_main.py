"""Small production mode for hosts that cannot keep the VAD checkpoints in RAM.

It keeps Mitsuka's public API shape for chat streaming and Piper speech while
using the configured Gemini backend.  The full ``main.py`` remains the local
desktop mode, where its 1.5 GB emotion checkpoints are appropriate.
"""
from __future__ import annotations

import asyncio
import json
import os
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from service.tts_service import TTSService

app = FastAPI(title="Mitsuka API (lite)")
voice_dir = Path(os.environ.get("PIPER_MODELS_DIR", "/opt/mitsuka/assets/models/voices"))
tts = TTSService(voice_dir)


class ChatRequest(BaseModel):
    query: str
    session_id: str = "default"


class SpeechRequest(BaseModel):
    input: str
    voice: str = "default"


def generate_reply(query: str) -> str:
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        return "Mitsuka đang chạy nhưng chưa có khoá Gemini để trả lời."
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=key)
    model = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
    contents = "Bạn là Mitsuka, một cô gái anime thân thiện. Trả lời ngắn gọn bằng tiếng Việt. " + query
    fresh_markers = ("đang hot", "gần đây", "mới nhất", "hôm nay", "thông tin mới")
    config = None
    if any(marker in query.casefold() for marker in fresh_markers):
        config = types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
        )

    try:
        result = client.models.generate_content(model=model, contents=contents, config=config)
    except Exception:
        # A custom/older Gemini model may not support Search grounding. The
        # proactive moment should still speak instead of failing the SSE turn.
        if config is None:
            raise
        result = client.models.generate_content(model=model, contents=contents)
    return result.text or "Mình chưa nghĩ ra câu trả lời phù hợp."


@app.get("/health")
@app.get("/health/live")
@app.get("/health/ready")
def health():
    return {"status": "ok", "mode": "lite"}


@app.post("/v1/chat/stream")
async def chat_stream(body: ChatRequest):
    turn_id = str(uuid.uuid4())

    async def events():
        yield ": ping\n\n"
        try:
            text = await asyncio.to_thread(generate_reply, body.query)
            yield "data: " + json.dumps({"type": "delta", "turn_id": turn_id, "content": text}, ensure_ascii=False) + "\n\n"
            yield "data: " + json.dumps({"type": "emotion", "turn_id": turn_id, "emotion": "calm", "state": {"valence": 0.55, "arousal": 0.45, "dominance": 0.5}}) + "\n\n"
            yield "data: " + json.dumps({"type": "done", "turn_id": turn_id, "agent_vad": {"valence": 0.55, "arousal": 0.45, "dominance": 0.5}}) + "\n\n"
        except Exception as exc:
            yield "data: " + json.dumps({"type": "error", "turn_id": turn_id, "error": {"code": "LLM_UNAVAILABLE", "message": str(exc), "retryable": True}}) + "\n\n"
            yield "data: " + json.dumps({"type": "done", "turn_id": turn_id}) + "\n\n"

    return StreamingResponse(events(), media_type="text/event-stream", headers={"X-Turn-Id": turn_id, "X-Accel-Buffering": "no"})


@app.get("/v1/audio/voices")
def voices():
    return {"voices": [voice["id"] for voice in tts.list_voices()]}


@app.post("/v1/audio/speech")
async def speech(body: SpeechRequest):
    if not body.input.strip():
        raise HTTPException(422, "input is empty")
    voice = "misuka-medium" if body.voice == "default" else body.voice
    if not tts.has_voice(voice):
        raise HTTPException(404, "voice not found")
    wav = await asyncio.to_thread(tts.synthesize_wav, voice, body.input)
    return Response(wav, media_type="audio/wav", headers={"Content-Disposition": "inline; filename=speech.wav"})


@app.post("/v1/audio/speech/finished", status_code=204)
def speech_finished():
    return Response(status_code=204)
