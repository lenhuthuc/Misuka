"""Shared pytest fixtures for the VAD/Brain FastAPI service.

`main.create_app()` never touches a real model or external backend at import
time — `ServiceContainer.create()` is the single seam where all of that
happens, and it only runs inside the FastAPI lifespan. So tests just
monkeypatch `ServiceContainer.create` to return an in-memory fake container;
no sys.modules faking or real network/model access is needed.
"""
from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import pytest_asyncio

VAD_ROOT = Path(__file__).resolve().parents[1]
if str(VAD_ROOT) not in sys.path:
    sys.path.insert(0, str(VAD_ROOT))

import main  # noqa: E402
from brain.emotion_service import EmotionService  # noqa: E402
from core.container import ServiceContainer  # noqa: E402
from core.tasks import BackgroundTaskRegistry  # noqa: E402
from core.tts_coordinator import TTSInterruptCoordinator  # noqa: E402


class FakeVADService:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def predict(self, text: str) -> tuple[float, float, float]:
        self.calls.append(text)
        return (0.0, 0.0, 0.0)


class FakeAudioEmotionService:
    def predict(self, audio, sample_rate: int = 16000) -> tuple[float, float, float]:
        return (0.1, 0.1, 0.1)


class FakeWhisperService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None]] = []  # (audio_path, language)

    def transcribe(self, audio_path: str, language: str | None = "en") -> str:
        self.calls.append((audio_path, language))
        return "fake transcript"


class _FakeVoice:
    def synthesize_wav(self, text: str, wf) -> None:
        # Large enough to span several of the API's 4096-byte stream chunks,
        # so interruption tests can observe a stream stopping mid-flight.
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 20000)


class FakeTTSService:
    _VOICE_ID = "fake-voice"

    def list_voices(self) -> list[dict]:
        return [{"id": self._VOICE_ID, "name": self._VOICE_ID}]

    def has_voice(self, voice_id: str) -> bool:
        return voice_id == self._VOICE_ID

    def get_voice(self, voice_id: str) -> _FakeVoice:
        return _FakeVoice()

    def get_sample_rate(self, voice_id: str) -> int:
        return 16000


class FakeLLMService:
    def __init__(
        self,
        response_text: str = "Xin chao, toi la tro ly ao.",
        stream_chunks: list[str] | None = None,
        stream_error: Exception | None = None,
        decide_answer: str = "NO",
    ) -> None:
        self.response_text = response_text
        self.stream_chunks = stream_chunks if stream_chunks is not None else ["Xin ", "chao", "!"]
        self.stream_error = stream_error
        self.decide_answer = decide_answer
        self.generate_calls: list[str] = []

    async def chat(self, messages) -> str:
        return self.response_text

    async def stream_chat(self, messages):
        if self.stream_error is not None:
            raise self.stream_error
        for chunk in self.stream_chunks:
            yield chunk

    async def generate(self, prompt: str) -> str:
        self.generate_calls.append(prompt)
        return self.decide_answer

    async def aclose(self) -> None:
        pass


class FakeMemoryService:
    def __init__(self) -> None:
        self.messages: list[dict] = []
        self.facts: dict[str, str] = {}

    async def get_recent(self, limit: int) -> list[dict]:
        return self.messages[-limit:]

    async def save_message(self, role: str, content: str, vad=None, emotion=None) -> None:
        self.messages.append({"role": role, "content": content, "vad": vad, "emotion": emotion})

    async def upsert_fact(self, key: str, value: str) -> None:
        self.facts[key] = value

    async def close(self) -> None:
        pass


class FakeVectorService:
    def __init__(self) -> None:
        self.upserted: list[tuple[list[str], list[dict]]] = []

    async def upsert(self, texts: list[str], metas: list[dict]) -> list[str]:
        ids = [str(i) for i in range(len(texts))]
        self.upserted.append((texts, metas))
        return ids

    async def close(self) -> None:
        pass


class FakeRAGService:
    def __init__(self, docs: list[dict] | None = None, context: str = "fake context") -> None:
        self.docs = docs if docs is not None else []
        self.context = context

    async def build_context(self, query: str):
        return [], self.docs, self.context


class FakeBrainBundle:
    """Everything ServiceContainer.create() would normally build, plus
    handles for assertions. Override fields in a test with e.g.
    `fake_brain_bundle.llm.stream_error = RuntimeError(...)`.
    """

    def __init__(self) -> None:
        self.vad = FakeVADService()
        self.audio_emotion = FakeAudioEmotionService()
        self.whisper = FakeWhisperService()
        self.tts = FakeTTSService()
        self.llm = FakeLLMService()
        self.memory = FakeMemoryService()
        self.vector = FakeVectorService()
        self.rag = FakeRAGService()

    def build_container(self) -> ServiceContainer:
        return ServiceContainer(
            vad=self.vad,
            audio_emotion=self.audio_emotion,
            whisper=self.whisper,
            tts=self.tts,
            llm=self.llm,
            memory=self.memory,
            vector=self.vector,
            rag=self.rag,
            emotion=EmotionService(self.vad),
            memory_recent_limit=10,
            emotion_executor=ThreadPoolExecutor(max_workers=2),
            tts_coordinator=TTSInterruptCoordinator(),
            tasks=BackgroundTaskRegistry(),
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
