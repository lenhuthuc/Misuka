from __future__ import annotations

import json
import logging
from typing import AsyncIterator

import httpx

logger = logging.getLogger(__name__)

_GENERATE_PATH = "/api/generate"
_CHAT_PATH = "/api/chat"

# Deliberation is selected per chat turn by brain.reasoning_policy. Although
# Ollama exposes Qwen3 thinking separately from content, the local fine-tune
# does not emit that trace and native thinking shares the answer's num_predict.
# A separate hidden pass below therefore owns its own hard token ceiling.

# How long Ollama keeps the weights resident after a request. Its default is
# five minutes, which is shorter than an ordinary pause in a conversation:
# measured on this box, a turn arriving after that pause paid 4.2s to reload
# qwen3:1.7b before it could read a single token of the prompt. The user is
# sitting in front of a companion that is supposed to answer in about a second,
# so that reload is most of the turn. Weights stay put instead -- 1.4GB against
# 13GB free, and nothing else on this machine is competing for it.
_KEEP_ALIVE = "30m"


_REASONING_SYSTEM = (
    "You are an internal reasoning module. Analyze the final user request using "
    "the supplied conversation and notes. State the likely interpretation, key "
    "reasoning, and conclusion compactly. Do not address the user or add social filler."
)


class LLMService:
    """Thin async wrapper around the Ollama REST API."""

    def __init__(
        self,
        base_url: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        reasoning_model: str = "qwen3:1.7b",
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_model = reasoning_model
        self._client = httpx.AsyncClient(base_url=self._base_url, timeout=120.0)

    async def generate(self, prompt: str, system: str = "") -> str:
        payload: dict = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature, "num_predict": self.max_tokens},
            # Background helpers (captioning/query classification) stay fast;
            # adaptive thinking is only meaningful for conversational turns.
            "think": False,
            "keep_alive": _KEEP_ALIVE,
        }
        if system:
            payload["system"] = system
        logger.debug("LLM generate | model=%s prompt_len=%d", self.model, len(prompt))
        resp = await self._client.post(_GENERATE_PATH, json=payload)
        resp.raise_for_status()
        return resp.json()["response"]

    async def reason(self, messages: list[dict[str, str]], max_tokens: int) -> str:
        """Run a hidden, hard-capped deliberation pass before the spoken answer.

        This deliberately uses non-thinking output from the stock reasoning
        model. Native Qwen3 thinking shares ``num_predict`` with its answer and
        may consume the entire ceiling; a separate concise pass makes the
        reasoning budget an actual hard bound.
        """
        reasoning_messages = [
            {"role": "system", "content": _REASONING_SYSTEM},
            *messages,
        ]
        payload = {
            "model": self.reasoning_model,
            "messages": reasoning_messages,
            "stream": False,
            "options": {"temperature": 0.2, "num_predict": max_tokens},
            "think": False,
            "keep_alive": _KEEP_ALIVE,
        }
        logger.info(
            "LLM reason | model=%s turns=%d token_budget=%d",
            self.reasoning_model, len(messages), max_tokens,
        )
        resp = await self._client.post(_CHAT_PATH, json=payload)
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    async def chat(
        self,
        messages: list[dict[str, str]],
        options: dict | None = None,
        *,
        think: bool = False,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": options or {"temperature": self.temperature, "num_predict": self.max_tokens},
            "think": think,
            "keep_alive": _KEEP_ALIVE,
        }
        logger.debug("LLM chat | model=%s turns=%d", self.model, len(messages))
        resp = await self._client.post(_CHAT_PATH, json=payload)
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    async def stream_chat(
        self,
        messages: list[dict[str, str]],
        options: dict | None = None,
        *,
        think: bool = False,
    ) -> AsyncIterator[str]:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": options or {"temperature": self.temperature, "num_predict": self.max_tokens},
            "think": think,
            "keep_alive": _KEEP_ALIVE,
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
