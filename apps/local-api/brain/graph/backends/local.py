"""Local backend: the fine-tuned Qwen3-1.7B (`mitsuka-ft`) served by Ollama.

The persona is in the weights and in the Modelfile's SYSTEM block, so `prepare`
returns the turn's messages untouched -- deliberately. Ollama injects that
SYSTEM only when the caller's first message is not itself a system message;
adding one here would silently replace the persona, and the model would stop
being Mitsuka (measured -- see `brain/nodes/generate.py`). For the same reason
few-shot examples are not sent: the fine-tune already carries that behaviour,
and re-teaching it every turn would only pay prefill to fight the training.

`keep_alive` is a latency number, not a tidiness one. Ollama's default unloads
the weights after five minutes, which is shorter than an ordinary pause in a
conversation; the reload measured 4.2s on this box, most of a turn Mitsuka is
supposed to answer in about one second.
"""
from __future__ import annotations

import logging
from collections.abc import AsyncIterator

import httpx
from langchain_ollama import ChatOllama

from brain.graph.backends.base import (
    BackendUnavailable,
    ChatBackend,
    Generation,
    GenerationOptions,
    Usage,
)

logger = logging.getLogger(__name__)

# Matches `brain.llm_service._KEEP_ALIVE`; both talk to the same runner.
_KEEP_ALIVE = "30m"


class LocalBackend(ChatBackend):
    name = "local"
    # A 1.7B is exactly the model the hidden deliberation pass was built for.
    supports_reasoning = True

    def __init__(
        self,
        base_url: str,
        model: str,
        temperature: float,
        max_tokens: int,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self._default = GenerationOptions(temperature=temperature, max_tokens=max_tokens)
        self._client = ChatOllama(
            model=model,
            base_url=base_url,
            # -> `"think": false` in the Ollama payload. The fine-tune reports a
            # thinking capability but emits no trace, and native thinking shares
            # num_predict with the answer, so it is off at the wire on every call.
            reasoning=False,
            keep_alive=_KEEP_ALIVE,
            temperature=temperature,
            num_predict=max_tokens,
            client_kwargs={"timeout": timeout},
        )

    def prepare(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        """Unchanged. The persona arrives from the Modelfile, not from here."""
        return messages

    @staticmethod
    def _options(options: GenerationOptions) -> dict[str, float | int]:
        # Only these two keys, matching `LLMService.chat`. Ollama request
        # options override Modelfile PARAMETERs, so anything named here would
        # silently take precedence over the values the model was tuned with.
        return {"temperature": options.temperature, "num_predict": options.max_tokens}

    @staticmethod
    def _read_usage(metadata: dict, usage: Usage) -> None:
        """Ollama reports counts under its own names, and not on every reply."""
        usage.input_tokens = int(metadata.get("prompt_eval_count") or 0)
        usage.output_tokens = int(metadata.get("eval_count") or 0)

    async def agenerate(
        self, messages: list[dict[str, str]], options: GenerationOptions
    ) -> Generation:
        try:
            reply = await self._client.ainvoke(
                self.prepare(messages), options=self._options(options)
            )
        except (httpx.HTTPError, ConnectionError, OSError) as exc:
            raise BackendUnavailable(f"ollama unreachable at generate: {exc}") from exc
        usage = Usage()
        self._read_usage(getattr(reply, "response_metadata", {}) or {}, usage)
        return Generation(text=_text_of(reply), usage=usage)

    async def astream(
        self,
        messages: list[dict[str, str]],
        options: GenerationOptions,
        usage: Usage,
    ) -> AsyncIterator[str]:
        try:
            async for chunk in self._client.astream(
                self.prepare(messages), options=self._options(options)
            ):
                metadata = getattr(chunk, "response_metadata", {}) or {}
                if metadata.get("done"):
                    self._read_usage(metadata, usage)
                if text := _text_of(chunk):
                    yield text
        except (httpx.HTTPError, ConnectionError, OSError) as exc:
            raise BackendUnavailable(f"ollama unreachable mid-stream: {exc}") from exc


def _text_of(message) -> str:
    """LangChain message content, whether it arrives as text or content blocks."""
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    text = getattr(message, "text", None)
    if callable(text):
        return text()
    return str(content or text or "")
