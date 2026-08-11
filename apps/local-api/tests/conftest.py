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

import pytest
import pytest_asyncio

VAD_ROOT = Path(__file__).resolve().parents[1]
if str(VAD_ROOT) not in sys.path:
    sys.path.insert(0, str(VAD_ROOT))

import main  # noqa: E402
from brain.curator import MemoryCurator  # noqa: E402
from brain.emotion_service import EmotionService  # noqa: E402
from core.container import ServiceContainer  # noqa: E402
from core.llm_priority import LLMPriorityGate  # noqa: E402
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

    def transcribe(
        self,
        audio_path: str,
        language: str | None = "en",
        prompt: str | None = None,
    ) -> str:
        self.calls.append((audio_path, language))
        return "fake transcript"


class FakeTTSService:
    _VOICE_ID = "fake-voice"

    def list_voices(self) -> list[dict]:
        return [{"id": self._VOICE_ID, "name": self._VOICE_ID, "engine": "fake"}]

    def has_voice(self, voice_id: str) -> bool:
        return voice_id == self._VOICE_ID

    def synthesize_wav(self, voice_id: str, text: str) -> bytes:
        # Large enough to span several of the API's 4096-byte stream chunks,
        # so interruption tests can observe a stream stopping mid-flight.
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

    async def chat(self, messages, options=None) -> str:
        self.chat_calls.append((messages, options))
        return self.response_text

    async def stream_chat(self, messages, options=None):
        self.stream_chat_calls.append((messages, options))
        if self.stream_error is not None:
            raise self.stream_error
        for chunk in self.stream_chunks:
            yield chunk

    async def generate(self, prompt: str) -> str:
        self.generate_calls.append(prompt)
        return self.generate_reply

    async def aclose(self) -> None:
        pass


class FakeMemoryService:
    def __init__(self) -> None:
        self.messages: list[dict] = []
        self.facts: dict[str, str] = {}
        self.curation_queue: list[dict] = []
        self._next_curation_id = 1

    async def get_recent(self, limit: int) -> list[dict]:
        return self.messages[-limit:]

    async def save_message(self, role: str, content: str, vad=None, emotion=None) -> None:
        # A timestamp is not incidental here: `prepare_turn` reads it to tell the
        # retriever which turns the history window already covers.
        self.messages.append({
            "role": role, "content": content, "vad": vad, "emotion": emotion,
            "timestamp": f"2026-08-09T00:00:{len(self.messages):02d}+00:00",
        })

    async def upsert_fact(self, key: str, value: str) -> None:
        self.facts[key] = value

    async def list_facts(self) -> list[dict]:
        return [{"key": k, "value": v} for k, v in self.facts.items()]

    async def enqueue_curation(self, query: str, response: str, emotion=None, vad=None) -> int:
        row_id = self._next_curation_id
        self._next_curation_id += 1
        v, a, d = vad if vad else (None, None, None)
        self.curation_queue.append({
            "id": row_id, "query": query, "response": response, "emotion": emotion,
            "valence": v, "arousal": a, "dominance": d,
            "created_at": "2026-08-09T00:00:00+00:00", "attempts": 0,
        })
        return row_id

    async def next_curation_batch(self, limit: int, max_attempts: int = 3) -> list[dict]:
        eligible = [r for r in self.curation_queue if r["attempts"] < max_attempts]
        return eligible[:limit]

    async def complete_curation(self, ids: list[int]) -> None:
        self.curation_queue = [r for r in self.curation_queue if r["id"] not in ids]

    async def record_curation_failure(self, ids: list[int]) -> None:
        for row in self.curation_queue:
            if row["id"] in ids:
                row["attempts"] += 1

    async def pending_curation_count(self, max_attempts: int = 3) -> int:
        return len([r for r in self.curation_queue if r["attempts"] < max_attempts])

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
        self.tts_default_voice = FakeTTSService._VOICE_ID
        self.llm = FakeLLMService()
        self.memory = FakeMemoryService()
        self.vector = FakeVectorService()
        self.rag = FakeRAGService()
        self.curator_llm = FakeLLMService()
        # No settle or quiet window: the suite asserts on routing, not on the
        # gate's timing, and a real quiet window would add 20s to every test
        # that touches a chat endpoint.
        self.llm_gate = LLMPriorityGate(settle_seconds=0.0, quiet_seconds=0.0)
        # Constructed but never started: routes only ever call `notify()` on it,
        # and a running drain loop would race every assertion in the suite.
        self.curator = MemoryCurator(
            memory=self.memory, llm=self.curator_llm, gate=self.llm_gate,
        )

    def build_container(self) -> ServiceContainer:
        return ServiceContainer(
            vad=self.vad,
            audio_emotion=self.audio_emotion,
            whisper=self.whisper,
            tts=self.tts,
            tts_default_voice=self.tts_default_voice,
            llm=self.llm,
            memory=self.memory,
            vector=self.vector,
            rag=self.rag,
            emotion=EmotionService(self.vad),
            memory_recent_limit=10,
            memory_recent_char_budget=3000,
            memory_facts_char_budget=600,
            emotion_executor=ThreadPoolExecutor(max_workers=2),
            tts_coordinator=TTSInterruptCoordinator(),
            tasks=BackgroundTaskRegistry(),
            # Near-zero settle so gated background work in tests does not add
            # real wall-clock delay to every chat assertion.
            llm_gate=self.llm_gate,
            curator=self.curator,
            curator_llm=self.curator_llm,
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
