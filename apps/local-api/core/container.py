"""Builds and tears down every service the app's routes depend on.

Replaces both the module-level singletons `main.py` used to construct at
import time (VAD model, audio-emotion model) and the ad-hoc
`brain.app.create_services`/`shutdown_services` pair — one factory, one
teardown, called once each from the FastAPI lifespan.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING

from brain.emotion_service import EmotionService
from core.tasks import BackgroundTaskRegistry
from core.tts_coordinator import TTSInterruptCoordinator
from model.vad_model import load_model
from service.audio_emotion_service import AudioEmotionService
from service.tts_service import TTSService
from service.vad_service import VADService
from service.whisper_service import WhisperService

if TYPE_CHECKING:
    from brain.config import Settings
    from brain.llm_service import LLMService
    from brain.memory_service import MemoryService
    from brain.rag_service import RAGService
    from brain.vector_service import VectorService

logger = logging.getLogger(__name__)


@dataclass
class ServiceContainer:
    vad: VADService
    audio_emotion: AudioEmotionService
    whisper: WhisperService
    tts: TTSService
    llm: "LLMService"
    memory: "MemoryService"
    vector: "VectorService"
    rag: "RAGService"
    emotion: EmotionService
    memory_recent_limit: int
    emotion_executor: ThreadPoolExecutor
    tts_coordinator: TTSInterruptCoordinator
    tasks: BackgroundTaskRegistry

    @classmethod
    async def create(cls, settings: "Settings") -> "ServiceContainer":
        from brain.embeddings import EmbeddingModel
        from brain.llm_service import LLMService
        from brain.memory_service import MemoryService
        from brain.rag_service import RAGService
        from brain.vector_service import VectorService

        vad_model, tokenizer = load_model(str(settings.resolved_vad_model_path))
        vad = VADService(vad_model, tokenizer)
        audio_emotion = AudioEmotionService()
        whisper = WhisperService(
            settings.whisper_model, settings.whisper_device, settings.whisper_compute,
            settings.whisper_models_dir,
        )
        tts = TTSService(settings.piper_models_dir)

        embedder = EmbeddingModel(settings.embedding_model_name, settings.embedding_batch_size)
        llm = LLMService(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )

        memory = MemoryService(settings.sqlite_path)
        await memory.initialize()

        vector = VectorService(
            embedding_model=embedder,
            collection_name=settings.qdrant_collection,
            qdrant_url=settings.qdrant_url,
            top_k=settings.qdrant_top_k,
        )
        await vector.initialize()

        rag = RAGService(vector=vector, rrf_k=settings.rag_rrf_k, top_k=settings.qdrant_top_k)
        emotion = EmotionService(vad)

        logger.info("Service container initialized")
        return cls(
            vad=vad,
            audio_emotion=audio_emotion,
            whisper=whisper,
            tts=tts,
            llm=llm,
            memory=memory,
            vector=vector,
            rag=rag,
            emotion=emotion,
            memory_recent_limit=settings.memory_recent_limit,
            emotion_executor=ThreadPoolExecutor(max_workers=settings.emotion_executor_max_workers),
            tts_coordinator=TTSInterruptCoordinator(),
            tasks=BackgroundTaskRegistry(),
        )

    async def shutdown(self) -> None:
        await self.tasks.drain()
        self.emotion_executor.shutdown(wait=True)
        await self.llm.aclose()
        await self.memory.close()
        await self.vector.close()
        logger.info("Service container shut down")
