from __future__ import annotations

import json
import logging
from typing import AsyncIterator

import httpx

logger = logging.getLogger(__name__)

_GENERATE_PATH = "/api/generate"
_CHAT_PATH = "/api/chat"

# Reasoning models (Qwen3 and up) emit a thinking pass before the answer, and
# Ollama bills it against `num_predict` like any other token. Measured on
# qwen3:4b with a 320-token ceiling, three of four spoken turns spent the whole
# budget reasoning in English and never reached a Vietnamese reply. Nothing here
# wants a visible reasoning trace — the output is read aloud — so it is off for
# every call. Ollama ignores the flag on models without the `thinking`
# capability (verified against qwen2.5: HTTP 200, no error), which keeps this
# safe to send unconditionally rather than gating it on the configured model.
_THINK = False


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
            "think": _THINK,
        }
        if system:
            payload["system"] = system
        logger.debug("LLM generate | model=%s prompt_len=%d", self.model, len(prompt))
        resp = await self._client.post(_GENERATE_PATH, json=payload)
        resp.raise_for_status()
        return resp.json()["response"]

    async def chat(self, messages: list[dict[str, str]], options: dict | None = None) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": options or {"temperature": self.temperature, "num_predict": self.max_tokens},
            "think": _THINK,
        }
        logger.debug("LLM chat | model=%s turns=%d", self.model, len(messages))
        resp = await self._client.post(_CHAT_PATH, json=payload)
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    async def stream_chat(self, messages: list[dict[str, str]], options: dict | None = None) -> AsyncIterator[str]:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": options or {"temperature": self.temperature, "num_predict": self.max_tokens},
            "think": _THINK,
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
