from __future__ import annotations

import asyncio
from collections import deque

import numpy as np

from brain.visual_memory import VisualMemory


class VisualBuffer:
    """Thread-safe ring buffer of VisualMemory entries (maxlen=20)."""

    def __init__(self, maxlen: int = 20) -> None:
        self._buf: deque[VisualMemory] = deque(maxlen=maxlen)
        self._lock = asyncio.Lock()

    async def push(self, vm: VisualMemory) -> None:
        async with self._lock:
            self._buf.append(vm)

    async def get_recent_visual_context(self, k: int = 5) -> list[VisualMemory]:
        async with self._lock:
            return list(self._buf)[-k:]

    def snapshot(self) -> list[VisualMemory]:
        """Non-blocking snapshot — safe to call from sync code."""
        return list(self._buf)

    async def search(self, query_embedding: np.ndarray, top_k: int = 3) -> list[VisualMemory]:
        """Return top_k entries by cosine similarity to query_embedding."""
        async with self._lock:
            items = list(self._buf)
        if not items:
            return []
        embeddings = np.stack([v.embedding for v in items])
        q = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / (norms + 1e-9)
        scores = normalized @ q
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [items[i] for i in top_indices]
