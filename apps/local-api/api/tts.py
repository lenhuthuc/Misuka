"""POST /v1/audio/speech -- one request, one complete spoken utterance.

An "utterance" is whatever the caller asks for, and the local conversation
client asks for one sentence at a time: waiting for the whole reply before
speaking any of it left the user listening to silence for the entire
generation (measured: 12s of decode after a 6s time-to-first-token, none of
it audible). Sentence-sized requests turn that into speech that starts once
the first sentence lands.

What made the earlier sentence-by-sentence attempt fail is fixed on both
sides rather than avoided:

- The boundary bug (a reply whose last sentence had no trailing whitespace
  was spoken once as the tail and again in full) was in the client's queue,
  and the client now tracks a consumed cursor instead -- see
  `local-conversation-tts.ts`.
- Prosody could not span a clause because the caller had no V/A/D to send
  yet: the reply's own reading only exists once the whole reply does. Hence
  `auto_prosody`, which lets this route read the sentence's emotion off the
  sentence itself. A sentence is a clause, which is exactly the span the
  Fujisaki contour is planned over (service/prosody.py).

`POST /v1/audio/speech/finished` closes the loop the other way: the client
reports when its playback queue drained, so the server knows the reply has
actually been heard. See core/llm_priority.py.
"""

import asyncio
import io
import logging
import wave
from dataclasses import replace

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response

from api.dependencies import get_container
from core.container import ServiceContainer
from core.logging import log_duration
from schemas.tts import TTSRequest
from service.prosody import ProsodyPlan, derive_prosody
from service.tts_service import TTSService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/audio", tags=["TTS"])


def wav_duration_seconds(wav_bytes: bytes) -> float:
    """How long `wav_bytes` takes to play, or 0.0 if it cannot be read.

    Use when: reserving conversation time on the LLM priority gate. A reply
    handed to the client is about to occupy the user for exactly this long,
    which is the window background work must stay out of.

    Returns 0.0 rather than raising: a malformed header is a reason to skip the
    reservation, never a reason to fail a synthesis request that succeeded.
    """
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            rate = wf.getframerate()
            return wf.getnframes() / rate if rate else 0.0
    except Exception:
        logger.debug("tts | could not read WAV duration for playback hold", exc_info=True)
        return 0.0


async def _synthesize_to_wav(
    tts: TTSService, voice_id: str, text: str, plan: ProsodyPlan | None,
) -> bytes:
    """Render off the event loop -- running Piper inline would stall every
    other request on the loop, including an in-flight chat stream.

    Timed because a reply is spoken a sentence at a time: the gap the user
    hears between two sentences is this call, and whether it is Piper or
    something else competing for the same cores is not a guess worth making.
    """
    with log_duration(logger, "tts_render", component="tts", chars=len(text)):
        return await asyncio.to_thread(tts.synthesize_wav, voice_id, text, plan)


async def _derive_vad_from_text(
    container: ServiceContainer, text: str,
) -> tuple[float, float, float] | None:
    """Read this utterance's own V/A/D off its words.

    The same text-only model `/v1/chat` runs over a finished reply
    (`analyze_agent_response`), pointed at one sentence instead. Runs on the
    emotion executor, like every other call into that pipeline, so a Piper
    render already in flight on another request keeps its own thread.

    Returns None on failure: a reply that speaks in the voice's default
    delivery is a far better outcome than one that does not speak at all.
    """
    try:
        loop = asyncio.get_event_loop()
        with log_duration(logger, "tts_auto_prosody", component="tts", chars=len(text)):
            return await loop.run_in_executor(
                container.emotion_executor, container.emotion_pipeline.analyze_agent_response, text,
            )
    except Exception:
        logger.exception("tts | could not derive prosody from text, using voice defaults")
        return None


def _resolve_voice(tts: TTSService, voice: str, configured_default: str) -> str:
    """Resolve a requested voice id, raising a stable error code instead of
    silently falling back to whichever voice happens to load first.

    "default" prefers the configured voice and only falls back to registry
    order when that one is not installed -- the fallback used to be the whole
    policy, which made the app's voice depend on filesystem ordering.
    """
    if voice == "default":
        voices = tts.list_voices()
        if not voices:
            raise HTTPException(
                status_code=503,
                detail={"code": "TTS_NO_VOICES_AVAILABLE", "message": "No TTS voice is installed."},
            )
        return configured_default if tts.has_voice(configured_default) else voices[0]["id"]

    if not tts.has_voice(voice):
        raise HTTPException(
            status_code=404,
            detail={"code": "TTS_UNKNOWN_VOICE", "message": f"Unknown voice '{voice}'."},
        )
    return voice


def _resolve_prosody(
    body: TTSRequest, container: ServiceContainer, vad: tuple[float, float, float] | None,
) -> ProsodyPlan | None:
    """Build the utterance's prosody plan, or None to use the voice defaults.

    `vad` is the reading to render with -- the caller's own, or one derived
    from the text under `auto_prosody`, or None when there is neither. It moves
    tempo alone; pitch comes from the voice's own configured brightness, the
    same for every utterance (see service/prosody.py).

    `speed` still wins where the caller set it explicitly: an OpenAI-compatible
    client asking for 1.5x speech means it, and having the emotion model
    quietly override that would make the parameter a lie.
    """
    if vad is None and abs(body.speed - 1.0) < 1e-3:
        return None

    plan = derive_prosody(
        *vad,
        depth=container.tts_prosody_depth,
        pitch_scale=container.tts_pitch_scale,
        contour_depth=container.tts_contour_depth,
    ) if vad else ProsodyPlan()
    if abs(body.speed - 1.0) >= 1e-3:
        # Piper's length_scale is inverse to speed: 2.0 is half as fast.
        plan = replace(plan, length_scale=min(max(plan.length_scale / body.speed, 0.3), 3.0))
    return plan


@router.get("/voices")
async def list_voices_endpoint(container: ServiceContainer = Depends(get_container)):
    return {"voices": [v["id"] for v in container.tts.list_voices()]}


@router.post("/speech")
async def synthesize(body: TTSRequest, container: ServiceContainer = Depends(get_container)):
    """Complete WAV for the whole reply, shaped by the agent's V/A/D."""
    if not body.input.strip():
        raise HTTPException(
            status_code=422,
            detail={"code": "TTS_EMPTY_INPUT", "message": "Trường 'input' rỗng."},
        )

    voice_id = _resolve_voice(container.tts, body.voice, container.tts_default_voice)
    vad = body.agent_vad
    if vad is None and body.auto_prosody:
        vad = await _derive_vad_from_text(container, body.input)
    plan = _resolve_prosody(body, container, vad)
    try:
        wav_bytes = await _synthesize_to_wav(container.tts, voice_id, body.input, plan)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail={"code": "TTS_SYNTHESIS_FAILED", "message": str(e)},
        ) from e

    container.llm_gate.hold_active(wav_duration_seconds(wav_bytes))

    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={"Content-Disposition": "inline; filename=speech.wav"},
    )


@router.post("/speech/finished", status_code=204)
async def speech_finished(container: ServiceContainer = Depends(get_container)) -> Response:
    """The client's playback queue has drained -- the reply has been heard.

    Use when: the last clip of a reply finished playing. One call per reply,
    not per clip.

    The durations reserved by `/speech` already estimate this moment, but only
    by assuming playback starts the instant synthesis returns. This is the
    client saying it actually happened, which is what releases the background
    work that was staying out of the way of playback (core/llm_priority.py).
    Unauthenticated and idempotent: the worst a stray call can do is let an
    embedding start early.
    """
    container.llm_gate.speech_finished()
    return Response(status_code=204)
