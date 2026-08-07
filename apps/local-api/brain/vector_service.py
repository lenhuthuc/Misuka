from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
    ScoredPoint,
)

from brain.embeddings import EmbeddingModel

logger = logging.getLogger(__name__)


class VectorService:
    """Qdrant-backed vector store with async search and batch support."""

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        collection_name: str = "brainmaster_docs",
        qdrant_url: str = "",
        top_k: int = 5,
    ) -> None:
        self._embedder = embedding_model
        self._collection = collection_name
        self._top_k = top_k
        # single async client — avoids dual in-memory instance mismatch
        self._client = AsyncQdrantClient(url=qdrant_url) if qdrant_url else AsyncQdrantClient(":memory:")

    async def initialize(self) -> None:
        dim = self._embedder.dimension
        existing = [c.name for c in (await self._client.get_collections()).collections]
        if self._collection not in existing:
            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
            logger.info("Created Qdrant collection '%s' dim=%d", self._collection, dim)
        else:
            logger.info("Qdrant collection '%s' already exists", self._collection)

    async def upsert(self, texts: list[str], metadatas: list[dict[str, Any]] | None = None) -> list[str]:
        vectors = await self._embedder.aembed(texts)
        metadatas = metadatas or [{} for _ in texts]
        ids = [str(uuid.uuid4()) for _ in texts]
        points = [
            PointStruct(
                id=pid,
                vector=vec.tolist(),
                payload={"content": txt, **meta},
            )
            for pid, vec, txt, meta in zip(ids, vectors, texts, metadatas)
        ]
        await self._client.upsert(collection_name=self._collection, points=points)
        logger.debug("Upserted %d points into '%s'", len(points), self._collection)
        return ids

    async def search(self, query: str, top_k: int | None = None) -> list[ScoredPoint]:
        vec = await self._embedder.aembed_one(query)
        response = await self._client.query_points(
            collection_name=self._collection,
            query=vec.tolist(),
            limit=top_k or self._top_k,
            with_payload=True,
        )
        return response.points

    async def search_batch(self, queries: list[str], top_k: int | None = None) -> list[list[ScoredPoint]]:
        tasks = [self.search(q, top_k) for q in queries]
        return await asyncio.gather(*tasks)

    async def close(self) -> None:
        await self._client.close()
