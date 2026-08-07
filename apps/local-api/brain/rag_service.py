from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING

from brain.state import RetrievedDoc

if TYPE_CHECKING:
    from brain.vector_service import VectorService

logger = logging.getLogger(__name__)


class RAGService:
    """Single-query RAG: searches vector store directly, fuses results via RRF."""

    def __init__(
        self,
        vector: "VectorService",
        rrf_k: int = 60,
        top_k: int = 5,
    ) -> None:
        self._vector = vector
        self._rrf_k = rrf_k
        self._top_k = top_k

    async def retrieve(self, query: str) -> tuple[list[str], list[RetrievedDoc]]:
        all_results = await self._vector.search_batch([query], top_k=self._top_k)
        docs = self._reciprocal_rank_fusion(all_results)
        logger.info("RAG | raw_hits=%d after_rrf=%d", sum(len(r) for r in all_results), len(docs))
        return [query], docs

    async def build_context(self, query: str) -> tuple[list[str], list[RetrievedDoc], str]:
        queries, docs = await self.retrieve(query)
        context = self._format_context(docs)
        return queries, docs, context

    def _reciprocal_rank_fusion(
        self,
        all_results: list[list],
    ) -> list[RetrievedDoc]:
        rrf_scores: dict[str, float] = defaultdict(float)
        payloads: dict[str, dict] = {}

        for results in all_results:
            for rank, hit in enumerate(results):
                doc_id = str(hit.id)
                rrf_scores[doc_id] += 1.0 / (self._rrf_k + rank + 1)
                if doc_id not in payloads:
                    payloads[doc_id] = hit.payload or {}

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [
            RetrievedDoc(
                doc_id=doc_id,
                content=payloads[doc_id].get("content", ""),
                score=score,
                metadata={k: v for k, v in payloads[doc_id].items() if k != "content"},
            )
            for doc_id, score in ranked[: self._top_k]
        ]

    @staticmethod
    def _format_context(docs: list[RetrievedDoc]) -> str:
        if not docs:
            return "No relevant documents found."
        lines = []
        for i, d in enumerate(docs):
            # Expose the stored emotion label so the agent consumes it directly
            # instead of re-deriving emotion from raw V/A/D numbers.
            emotion = (d.get("metadata") or {}).get("emotion")
            tag = f" (felt: {emotion})" if emotion else ""
            lines.append(f"[{i+1}]{tag} {d['content']}")
        return "\n\n".join(lines)
