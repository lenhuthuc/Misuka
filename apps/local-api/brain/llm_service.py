from __future__ import annotations

import json
import logging
from typing import AsyncIterator

import httpx

logger = logging.getLogger(__name__)

_GENERATE_PATH = "/api/generate"
_CHAT_PATH = "/api/chat"


class LLMService:
    """Thin async wrapper around the Ollama REST API."""

    def __init__(self, base_url: str, model: str, temperature: float = 0.7, max_tokens: int = 1024) -> None:
        self._base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = httpx.AsyncClient(base_url=self._base_url, timeout=120.0)

    async def generate(self, prompt: str, system: str = "") -> str:
        payload: dict = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature, "num_predict": self.max_tokens},
        }
        if system:
            payload["system"] = system
        logger.debug("LLM generate | model=%s prompt_len=%d", self.model, len(prompt))
        resp = await self._client.post(_GENERATE_PATH, json=payload)
        resp.raise_for_status()
        return resp.json()["response"]

    async def chat(self, messages: list[dict[str, str]]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": self.temperature, "num_predict": self.max_tokens},
        }
        logger.debug("LLM chat | model=%s turns=%d", self.model, len(messages))
        resp = await self._client.post(_CHAT_PATH, json=payload)
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    async def stream_chat(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": self.temperature, "num_predict": self.max_tokens},
        }
        async with self._client.stream("POST", _CHAT_PATH, json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line:
                    data = json.loads(line)
                    if chunk := data.get("message", {}).get("content", ""):
                        yield chunk
                    if data.get("done"):
                        break

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "LLMService":
        return self

    async def __aexit__(self, *_) -> None:
        await self.aclose()
