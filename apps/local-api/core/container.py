"""Builds and tears down every service the app's routes depend on.

Replaces both the module-level singletons `main.py` used to construct at
import time (VAD model, audio-emotion model) and the ad-hoc
`brain.app.create_services`/`shutdown_services` pair — one factory, one
teardown, called once each from the FastAPI lifespan.
"""
from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING

from brain.caption_service import CaptionService
from brain.emotion_service import EmotionService
from brain.graph import GraphConfig, GraphDeps, MitsukaGraph
from brain.graph.backends import BackendRegistry, CloudAvailability, CloudBackend, LocalBackend
from brain.graph.guard import GuardConfig
from brain.graph.metrics import TurnMetrics
from brain.graph.prompts import warn_on_persona_drift
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
    from brain.web_search_service import WebSearchService

logger = logging.getLogger(__name__)


def build_conversation_graph(
    settings: "Settings",
    *,
    llm: "LLMService",
    memory: "MemoryService",
    rag: "RAGService",
    web_search: "WebSearchService",
    gate: LLMPriorityGate,
    tasks: BackgroundTaskRegistry,
) -> MitsukaGraph:
    """Assemble the LangGraph turn pipeline from settings.

    Kept a module-level function rather than inlined in `create` so the demo
    script in `scripts/` can build the same graph over the same settings
    without standing up ASR, TTS and three ONNX models to do it.
    """
    local = LocalBackend(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )
    cloud: CloudBackend | None = None
    if settings.cloud_enabled and settings.gemini_api_key:
        cloud = CloudBackend(
            api_key=settings.gemini_api_key,
            model=settings.gemini_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            timeout=settings.cloud_timeout_seconds,
            max_retries=settings.cloud_max_retries,
        )
        # Only meaningful when a cloud backend exists: it is the one that sends
        # the persona as text, so it is the one drift can silence.
        warn_on_persona_drift()
    else:
        logger.info(
            "conversation graph | cloud backend disabled (%s)",
            "no GEMINI_API_KEY" if settings.cloud_enabled else "cloud_enabled=false",
        )

    registry = BackendRegistry(
        local=local,
        cloud=cloud,
        availability=CloudAvailability(
            transient_cooldown_seconds=settings.cloud_transient_cooldown_seconds,
            quota_reset_timezone=settings.cloud_quota_reset_timezone,
        ),
        prefer_cloud=settings.cloud_enabled,
    )

    config = GraphConfig(
        history_turns=settings.graph_history_turns,
        summary_max_tokens=settings.graph_summary_max_tokens,
        summary_defer_timeout=settings.graph_summary_defer_timeout,
        memory_recent_limit=settings.memory_recent_limit,
        memory_recent_char_budget=settings.memory_recent_char_budget,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        reasoning_enabled=settings.reasoning_enabled,
        reasoning_activation_threshold=settings.reasoning_activation_threshold,
        reasoning_min_tokens=settings.reasoning_min_tokens,
        reasoning_max_tokens=settings.reasoning_max_tokens,
        reasoning_token_scale=settings.reasoning_token_scale,
        web_search_enabled=settings.web_search_enabled,
        web_search_knowledge_enabled=settings.web_search_knowledge_enabled,
        knowledge_temperature=settings.knowledge_temperature,
        knowledge_max_tokens=settings.knowledge_max_tokens,
        guard=GuardConfig(
            max_sentences=settings.guard_max_sentences,
            max_chars=settings.guard_max_chars,
            previous_turn_similarity=settings.guard_previous_turn_similarity,
        ),
        max_regenerations=settings.graph_max_regenerations,
    )

    deps = GraphDeps(
        memory=memory,
        rag=rag,
        backends=registry,
        metrics=TurnMetrics(
            settings.turn_metrics_path, enabled=settings.turn_metrics_enabled
        ),
        config=config,
        llm=llm,
        web_search=web_search,
        gate=gate,
        tasks=tasks,
    )
    return MitsukaGraph(deps)


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
    web_search: "WebSearchService"
    web_search_enabled: bool
    web_search_knowledge_enabled: bool
    knowledge_temperature: float
    knowledge_max_tokens: int
    emotion: EmotionService
    # None when `vision_captioning_enabled` is off -- callers must treat that
    # the same as "captioning produced nothing" rather than a missing service.
    caption: CaptionService | None
    vision_caption_timeout_seconds: float
    memory_recent_limit: int
    memory_recent_char_budget: int
    reasoning_enabled: bool
    reasoning_activation_threshold: float
    reasoning_min_tokens: int
    reasoning_max_tokens: int
    reasoning_token_scale: float
    emotion_executor: ThreadPoolExecutor
    tasks: BackgroundTaskRegistry
    llm_gate: LLMPriorityGate
    # The LangGraph turn pipeline. Both chat endpoints run through this; `llm`
    # above is still here because the graph's local backend is not the only
    # caller of Ollama (captioning and the deliberation pass use it directly).
    graph: MitsukaGraph
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
        from brain.web_search_service import WebSearchService

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
        # Piper's first ONNX load used to occur after the first model response,
        # turning a ready reply into a multi-second silent wait. Pay it during
        # service startup so the first streamed sentence can speak immediately.
        if tts.has_voice(settings.tts_default_voice):
            await asyncio.to_thread(tts.preload, settings.tts_default_voice)

        embedder = EmbeddingModel(settings.embedding_model_name, settings.embedding_batch_size)
        llm = LLMService(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            reasoning_model=settings.reasoning_model,
        )

        memory = MemoryService(
            settings.sqlite_path,
            bm25_repetition_threshold=settings.bm25_repetition_threshold,
            bm25_repetition_min_tokens=settings.bm25_repetition_min_tokens,
            bm25_repetition_max_sentences=settings.bm25_repetition_max_sentences,
        )
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
        web_search = WebSearchService(
            max_results=settings.web_search_max_results,
            context_char_budget=settings.web_search_context_char_budget,
            timeout_seconds=settings.web_search_timeout_seconds,
            region=settings.web_search_region,
        )
        emotion = EmotionService(text_vad)

        caption: CaptionService | None = None
        if settings.vision_captioning_enabled:
            # `CaptionService.__init__` builds the `vision/` pipeline, which
            # loads three ONNX graphs off disk (the CLIP towers are ~900MB
            # between them) -- off the event loop like every other model
            # construction here, so it can't stall startup for routes that
            # don't touch vision.
            loop = asyncio.get_event_loop()
            caption = await loop.run_in_executor(None, CaptionService)

        llm_gate = LLMPriorityGate(speech_lull_seconds=settings.llm_speech_lull_seconds)
        tasks = BackgroundTaskRegistry()

        graph = build_conversation_graph(
            settings,
            llm=llm,
            memory=memory,
            rag=rag,
            web_search=web_search,
            gate=llm_gate,
            tasks=tasks,
        )

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
            web_search=web_search,
            web_search_enabled=settings.web_search_enabled,
            web_search_knowledge_enabled=settings.web_search_knowledge_enabled,
            knowledge_temperature=settings.knowledge_temperature,
            knowledge_max_tokens=settings.knowledge_max_tokens,
            emotion=emotion,
            caption=caption,
            vision_caption_timeout_seconds=settings.vision_caption_timeout_seconds,
            memory_recent_limit=settings.memory_recent_limit,
            memory_recent_char_budget=settings.memory_recent_char_budget,
            reasoning_enabled=settings.reasoning_enabled,
            reasoning_activation_threshold=settings.reasoning_activation_threshold,
            reasoning_min_tokens=settings.reasoning_min_tokens,
            reasoning_max_tokens=settings.reasoning_max_tokens,
            reasoning_token_scale=settings.reasoning_token_scale,
            emotion_executor=ThreadPoolExecutor(max_workers=settings.emotion_executor_max_workers),
            tasks=tasks,
            llm_gate=llm_gate,
            graph=graph,
            index_defer_timeout=settings.llm_speech_defer_seconds,
        )

    async def shutdown(self) -> None:
        await self.tasks.drain()
        self.emotion_executor.shutdown(wait=True)
        await self.llm.aclose()
        await self.memory.close()
        await self.vector.close()
        logger.info("Service container shut down")
