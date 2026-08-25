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
from core.llm_priority import LLMPriorityGate
from core.tasks import BackgroundTaskRegistry
from model.multimodal_vad import load_multimodal_vad
from model.text_vad import load_text_vad
from service.emotion_pipeline import EmotionPipeline
from service.multimodal_vad_service import MultimodalVADService
from service.sherpa_asr_service import SherpaASRService
from service.text_vad_service import TextVADService
from service.tts_service import TTSService

if TYPE_CHECKING:
    from brain.config import Settings
    from brain.llm_service import LLMService
    from brain.memory_service import MemoryService
    from brain.rag_service import RAGService
    from brain.vector_service import VectorService

logger = logging.getLogger(__name__)


@dataclass
class ServiceContainer:
    text_vad: TextVADService
    multimodal_vad: MultimodalVADService
    asr: SherpaASRService
    emotion_pipeline: EmotionPipeline
    tts: TTSService
    tts_default_voice: str
    tts_prosody_depth: float
    tts_pitch_scale: float
    tts_contour_depth: float
    llm: "LLMService"
    memory: "MemoryService"
    vector: "VectorService"
    rag: "RAGService"
    emotion: EmotionService
    memory_recent_limit: int
    memory_recent_char_budget: int
    emotion_executor: ThreadPoolExecutor
    tasks: BackgroundTaskRegistry
    llm_gate: LLMPriorityGate
    # How long the exchange-indexing task waits for the reply to finish being
    # spoken before giving up on the signal and embedding anyway.
    index_defer_timeout: float = 90.0

    @classmethod
    async def create(cls, settings: "Settings") -> "ServiceContainer":
        from brain.embeddings import EmbeddingModel
        from brain.llm_service import LLMService
        from brain.memory_service import MemoryService
        from brain.rag_service import RAGService
        from brain.vector_service import VectorService

        text_vad_model, text_vad_tokenizer = load_text_vad(str(settings.resolved_text_vad_checkpoint_path))
        text_vad = TextVADService(text_vad_model, text_vad_tokenizer)

        multimodal_vad_model, multimodal_vad_tokenizer = load_multimodal_vad(
            str(settings.resolved_multimodal_vad_checkpoint_path)
        )
        multimodal_vad = MultimodalVADService(multimodal_vad_model, multimodal_vad_tokenizer)

        asr = SherpaASRService(
            tokens=settings.resolved_sherpa_tokens,
            encoder=settings.resolved_sherpa_encoder,
            decoder=settings.resolved_sherpa_decoder,
            joiner=settings.resolved_sherpa_joiner,
            num_threads=settings.sherpa_onnx_num_threads,
        )
        emotion_pipeline = EmotionPipeline(asr, multimodal_vad, text_vad)

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

        rag = RAGService(
            vector=vector,
            rrf_k=settings.rag_rrf_k,
            top_k=settings.qdrant_top_k,
            context_char_budget=settings.rag_context_char_budget,
            min_score=settings.rag_min_score,
        )
        emotion = EmotionService(text_vad)

        llm_gate = LLMPriorityGate(speech_lull_seconds=settings.llm_speech_lull_seconds)

        logger.info("Service container initialized")
        return cls(
            text_vad=text_vad,
            multimodal_vad=multimodal_vad,
            asr=asr,
            emotion_pipeline=emotion_pipeline,
            tts=tts,
            tts_default_voice=settings.tts_default_voice,
            tts_prosody_depth=settings.tts_prosody_depth,
            tts_pitch_scale=settings.tts_pitch_scale,
            tts_contour_depth=settings.tts_contour_depth,
            llm=llm,
            memory=memory,
            vector=vector,
            rag=rag,
            emotion=emotion,
            memory_recent_limit=settings.memory_recent_limit,
            memory_recent_char_budget=settings.memory_recent_char_budget,
            emotion_executor=ThreadPoolExecutor(max_workers=settings.emotion_executor_max_workers),
            tasks=BackgroundTaskRegistry(),
            llm_gate=llm_gate,
            index_defer_timeout=settings.llm_speech_defer_seconds,
        )

    async def shutdown(self) -> None:
        await self.tasks.drain()
        self.emotion_executor.shutdown(wait=True)
        await self.llm.aclose()
        await self.memory.close()
        await self.vector.close()
        logger.info("Service container shut down")
