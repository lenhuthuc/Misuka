"""Shared pytest fixtures for the VAD/Brain FastAPI service.

`main.create_app()` never touches a real model or external backend at import
time — `ServiceContainer.create()` is the single seam where all of that
happens, and it only runs inside the FastAPI lifespan. So tests just
monkeypatch `ServiceContainer.create` to return an in-memory fake container;
no sys.modules faking or real network/model access is needed.
"""
from __future__ import annotations

import io
import sys
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import pytest_asyncio

VAD_ROOT = Path(__file__).resolve().parents[1]
if str(VAD_ROOT) not in sys.path:
    sys.path.insert(0, str(VAD_ROOT))

import main  # noqa: E402
from brain.emotion_service import EmotionService  # noqa: E402
from core.container import ServiceContainer  # noqa: E402
from core.llm_priority import LLMPriorityGate  # noqa: E402
from core.tasks import BackgroundTaskRegistry  # noqa: E402
from service.emotion_pipeline import EmotionPipeline  # noqa: E402


class FakeTextVADService:
    """Stands in for `service.text_vad_service.TextVADService`."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def predict(self, text: str) -> tuple[float, float, float]:
        """Signed [-1, 1] — what `EmotionService`/`/vad` expect."""
        self.calls.append(text)
        return (0.0, 0.0, 0.0)

    def predict_signed(self, text: str) -> tuple[float, float, float]:
        return self.predict(text)

    def predict_raw(self, text: str) -> tuple[float, float, float]:
        """[0, 1] — the checkpoint's native range, used for `agent_vad`."""
        self.calls.append(text)
        return (0.5, 0.5, 0.5)


class FakeMultimodalVADService:
    """Stands in for `service.multimodal_vad_service.MultimodalVADService`."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, int, str]] = []  # (len(audio), len(mask), text)

    def predict(self, audio, audio_attention_mask, text: str) -> tuple[float, float, float]:
        self.calls.append((len(audio), len(audio_attention_mask), text))
        return (0.6, 0.6, 0.6)


class FakeASRService:
    """Stands in for `service.sherpa_asr_service.SherpaASRService`."""

    def __init__(self) -> None:
        self.calls: list[int] = []  # len(samples) per call

    def transcribe(self, samples) -> str:
        self.calls.append(len(samples))
        return "fake transcript"


class FakeCaptionService:
    """Stands in for `brain.caption_service.CaptionService`."""

    def __init__(self) -> None:
        self.calls: list[np.ndarray] = []
        self.caption_to_return = "a fake image caption"

    async def caption(self, frame: np.ndarray) -> str:
        self.calls.append(frame)
        return self.caption_to_return


class FakeTTSService:
    _VOICE_ID = "fake-voice"

    def list_voices(self) -> list[dict]:
        return [{"id": self._VOICE_ID, "name": self._VOICE_ID, "engine": "fake"}]

    def has_voice(self, voice_id: str) -> bool:
        return voice_id == self._VOICE_ID

    def synthesize_wav(self, voice_id: str, text: str, plan: object | None = None) -> bytes:
        # Recorded rather than used: the route decides *whether* a prosody plan
        # exists at all (V/A/D present, `speed` overridden), and that decision
        # is what the tests assert on.
        self.last_plan = plan
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"\x00\x00" * 20000)
        return buf.getvalue()


class FakeLLMService:
    def __init__(
        self,
        response_text: str = "Xin chao, toi la tro ly ao.",
        stream_chunks: list[str] | None = None,
        stream_error: Exception | None = None,
        generate_reply: str = "NO",
    ) -> None:
        self.temperature = 0.7
        self.max_tokens = 1024
        self.response_text = response_text
        self.stream_chunks = stream_chunks if stream_chunks is not None else ["Xin ", "chao", "!"]
        self.stream_error = stream_error
        self.generate_reply = generate_reply
        self.generate_calls: list[str] = []
        self.chat_calls: list[tuple[list[dict], dict | None]] = []
        self.stream_chat_calls: list[tuple[list[dict], dict | None]] = []
        self.chat_think_calls: list[bool] = []
        self.stream_chat_think_calls: list[bool] = []
        self.reason_calls: list[tuple[list[dict], int]] = []

    async def chat(self, messages, options=None, *, think: bool = False) -> str:
        self.chat_calls.append((messages, options))
        self.chat_think_calls.append(think)
        return self.response_text

    async def stream_chat(self, messages, options=None, *, think: bool = False):
        self.stream_chat_calls.append((messages, options))
        self.stream_chat_think_calls.append(think)
        if self.stream_error is not None:
            raise self.stream_error
        for chunk in self.stream_chunks:
            yield chunk

    async def generate(self, prompt: str) -> str:
        self.generate_calls.append(prompt)
        return self.generate_reply

    async def reason(self, messages, max_tokens: int) -> str:
        self.reason_calls.append((messages, max_tokens))
        return "internal analysis and conclusion"

    async def aclose(self) -> None:
        pass


class FakeMemoryService:
    def __init__(self) -> None:
        self.messages: list[dict] = []

    async def get_recent(self, limit: int) -> list[dict]:
        return self.messages[-limit:]

    async def save_message(self, role: str, content: str, vad=None, emotion=None) -> None:
        # A timestamp is not incidental here: `prepare_turn` reads it to tell the
        # retriever which turns the history window already covers.
        self.messages.append({
            "role": role, "content": content, "vad": vad, "emotion": emotion,
            "timestamp": f"2026-08-09T00:00:{len(self.messages):02d}+00:00",
        })

    async def close(self) -> None:
        pass


class FakeVectorService:
    def __init__(self) -> None:
        self.upserted: list[tuple[list[str], list[dict]]] = []
        self._next_id = 0

    async def upsert(self, texts: list[str], metas: list[dict]) -> list[str]:
        ids = [f"point-{self._next_id + i}" for i in range(len(texts))]
        self._next_id += len(texts)
        self.upserted.append((texts, metas))
        return ids

    async def close(self) -> None:
        pass


class FakeRAGService:
    def __init__(self, docs: list[dict] | None = None, context: str = "fake context") -> None:
        self.docs = docs if docs is not None else []
        self.context = context
        # Records what the turn said its history window already covers, so tests
        # can assert the retriever is told to skip duplicated turns.
        self.covered_since_calls: list[str | None] = []

    async def build_context(self, query: str, covered_since: str | None = None):
        self.covered_since_calls.append(covered_since)
        return [], self.docs, self.context


class FakeWebSearchService:
    def __init__(self, results: list[dict] | None = None, context: str = "") -> None:
        self.results = results if results is not None else []
        self.context = context
        self.error: Exception | None = None
        self.query_calls: list[str] = []

    async def build_context(self, query: str):
        self.query_calls.append(query)
        if self.error is not None:
            raise self.error
        return self.results, self.context


class FakeBrainBundle:
    """Everything ServiceContainer.create() would normally build, plus
    handles for assertions. Override fields in a test with e.g.
    `fake_brain_bundle.llm.stream_error = RuntimeError(...)`.
    """

    def __init__(self) -> None:
        self.text_vad = FakeTextVADService()
        self.multimodal_vad = FakeMultimodalVADService()
        self.asr = FakeASRService()
        # Real orchestrator over the fakes above — exercises the actual
        # decode/resample/center-crop pipeline (service/audio_preprocessing.py)
        # against real WAV bytes, only the two models + ASR are faked.
        self.emotion_pipeline = EmotionPipeline(self.asr, self.multimodal_vad, self.text_vad)
        self.tts = FakeTTSService()
        self.tts_default_voice = FakeTTSService._VOICE_ID
        self.tts_prosody_depth = 1.0
        self.tts_pitch_scale = 1.06
        self.tts_contour_depth = 0.0
        self.llm = FakeLLMService()
        self.memory = FakeMemoryService()
        self.vector = FakeVectorService()
        self.rag = FakeRAGService()
        self.web_search = FakeWebSearchService()
        self.web_search_enabled = True
        self.web_search_knowledge_enabled = True
        self.caption = FakeCaptionService()
        self.vision_caption_timeout_seconds = 30.0
        self.llm_gate = LLMPriorityGate()

    def build_container(self) -> ServiceContainer:
        return ServiceContainer(
            text_vad=self.text_vad,
            multimodal_vad=self.multimodal_vad,
            asr=self.asr,
            emotion_pipeline=self.emotion_pipeline,
            tts=self.tts,
            tts_default_voice=self.tts_default_voice,
            tts_prosody_depth=self.tts_prosody_depth,
            tts_pitch_scale=self.tts_pitch_scale,
            tts_contour_depth=self.tts_contour_depth,
            llm=self.llm,
            memory=self.memory,
            vector=self.vector,
            rag=self.rag,
            web_search=self.web_search,
            web_search_enabled=self.web_search_enabled,
            web_search_knowledge_enabled=self.web_search_knowledge_enabled,
            knowledge_temperature=0.30,
            knowledge_max_tokens=480,
            emotion=EmotionService(self.text_vad),
            caption=self.caption,
            vision_caption_timeout_seconds=self.vision_caption_timeout_seconds,
            memory_recent_limit=18,
            memory_recent_char_budget=3000,
            reasoning_enabled=True,
            reasoning_activation_threshold=0.50,
            reasoning_min_tokens=64,
            reasoning_max_tokens=192,
            reasoning_token_scale=0.65,
            emotion_executor=ThreadPoolExecutor(max_workers=2),
            tasks=BackgroundTaskRegistry(),
            llm_gate=self.llm_gate,
        )


@pytest.fixture
def fake_brain_bundle() -> FakeBrainBundle:
    """Must be requested *before* `client` in a test's parameter list so the
    same instance is wired into the app (fixtures build lazily, in argument order).
    """
    return FakeBrainBundle()


@pytest_asyncio.fixture
async def client(monkeypatch, fake_brain_bundle: FakeBrainBundle):
    """An httpx.AsyncClient bound to the real ASGI app via in-process transport.

    Drives the app's actual lifespan (startup/shutdown) so tests exercise the
    same wiring as production, with only ServiceContainer.create swapped out.
    """
    import httpx

    async def _fake_create(cls, settings):
        return fake_brain_bundle.build_container()

    monkeypatch.setattr(ServiceContainer, "create", classmethod(_fake_create))

    async with main.app.router.lifespan_context(main.app):
        # raise_app_exceptions=False: Starlette's ServerErrorMiddleware sends
        # the client-visible response from our catch-all exception handler
        # *and* re-raises for the ASGI server's own error logging (uvicorn
        # swallows that re-raise in production). httpx's default of re-raising
        # it into the test would hide the response we actually want to assert on.
        transport = httpx.ASGITransport(app=main.app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
            yield ac


def make_wav_bytes(duration_sec: float = 0.5, sample_rate: int = 16000) -> bytes:
    """Silent mono 16-bit WAV — enough for endpoints that only need valid audio framing."""
    import io
    import wave

    n_samples = int(duration_sec * sample_rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_samples)
    return buf.getvalue()
