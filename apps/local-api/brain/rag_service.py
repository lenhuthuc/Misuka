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
        context_char_budget: int = 2000,
    ) -> None:
        self._vector = vector
        self._rrf_k = rrf_k
        self._top_k = top_k
        self._context_char_budget = context_char_budget

    async def retrieve(self, query: str) -> tuple[list[str], list[RetrievedDoc]]:
        all_results = await self._vector.search_batch([query], top_k=self._top_k)
        docs = self._reciprocal_rank_fusion(all_results)
        logger.info("RAG | raw_hits=%d after_rrf=%d", sum(len(r) for r in all_results), len(docs))
        return [query], docs

    async def build_context(
        self,
        query: str,
        covered_since: str | None = None,
    ) -> tuple[list[str], list[RetrievedDoc], str]:
        """Retrieve long-term context for a turn.

        Use when: building the system prompt for a chat turn that will also
        carry a verbatim recent-history window.

        Expects: `covered_since` is the ISO timestamp of the oldest message
        already in that history window. Every exchange is indexed into the
        vector store as it happens, so without this the retriever hands back
        the same recent turns the history already contains — measured on a real
        five-turn conversation, that duplication grew the prompt from 4.5k to
        15k characters and time-to-first-token from 11s to 44s.

        Returns: the queries issued, the docs that survived filtering and the
        character budget, and those docs formatted as prompt context.
        """
        queries, docs = await self.retrieve(query)

        if covered_since is not None:
            docs = [d for d in docs if not self._already_in_history(d, covered_since)]

        docs = self._fit_budget(docs)
        return queries, docs, self._format_context(docs)

    @staticmethod
    def _already_in_history(doc: RetrievedDoc, covered_since: str) -> bool:
        """Docs indexed at or after the history window's oldest message are, by
        construction, turns that window already spells out verbatim."""
        timestamp = (doc.get("metadata") or {}).get("timestamp")
        return bool(timestamp) and timestamp >= covered_since

    def _fit_budget(self, docs: list[RetrievedDoc]) -> list[RetrievedDoc]:
        """Keep docs in rank order until the character budget is spent.

        Prefill cost is linear in prompt length and dominates time-to-first-token
        on a CPU runner, so an unbounded context trades latency for recall the
        user never asked for. Stored responses grow without limit, hence a
        character budget rather than a document count.
        """
        kept: list[RetrievedDoc] = []
        spent = 0
        for doc in docs:
            cost = len(doc["content"])
            if kept and spent + cost > self._context_char_budget:
                break
            kept.append(doc)
            spent += cost

        if len(kept) < len(docs):
            logger.info(
                "RAG | context budget trimmed %d -> %d docs (%d chars)",
                len(docs), len(kept), spent,
            )
        return kept

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
