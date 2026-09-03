"""Cloud backend: Gemini via Google AI Studio's free tier.

Gemini has no Modelfile, so this is where the persona has to be spoken aloud:
`prepare` puts `prompts/system_prompt.txt` -- byte-identical to the SYSTEM block
`mitsuka-ft` was built with, enforced by `prompts.check_persona_drift` -- at the
head of every call, followed by the few-shot exchanges the local backend does
not need because they are in its weights.

The model name is config, never a literal. A model withdrawn from an account is
also a routing failure: local generation remains available while configuration
is corrected instead of turning the user's turn into a 5xx.
"""
from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator

import httpx
from google.genai import errors as genai_errors
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import (
    GoogleInvalidRequestError,
    GoogleModelNotFoundError,
)

from brain.graph.backends.base import (
    BackendUnavailable,
    ChatBackend,
    DailyQuotaExceeded,
    Generation,
    GenerationOptions,
    Usage,
)
from brain.graph.prompts import load_fewshot, load_system_prompt

logger = logging.getLogger(__name__)

# Same header the local path uses for retrieved material, so the untrusted-span
# warning reads identically on both backends.
_NOTE_HEADER = "Ghi chú cho lượt này:"


def _is_daily_quota(exc: genai_errors.APIError) -> bool:
    """Tell today's allowance being gone from a per-minute burst limit.

    Both are 429. Only the daily one should park the cloud route for hours --
    treating a per-minute limit that way would hand the rest of the day to a
    1.7B over a burst that clears in sixty seconds. Google names the quota in
    the error's QuotaFailure details (`...PerDay...`), so this reads the
    failure rather than assuming from the status code.
    """
    haystack = f"{getattr(exc, 'message', '')} {json.dumps(getattr(exc, 'details', None), default=str)}"
    lowered = haystack.lower()
    return "perday" in lowered or "per day" in lowered


def _translate(exc: Exception) -> Exception:
    """Map a provider error onto the routing taxonomy in `backends.base`."""
    if isinstance(exc, (GoogleInvalidRequestError, GoogleModelNotFoundError)):
        # Gemini 3 model/API incompatibilities are surfaced by LangChain as
        # this wrapper rather than google.genai.errors.APIError. A local reply
        # is still preferable to exposing its 400 as a server error.
        return BackendUnavailable(f"gemini rejected this request: {exc}")
    if isinstance(exc, genai_errors.APIError):
        if exc.code == 429:
            if _is_daily_quota(exc):
                return DailyQuotaExceeded(f"gemini daily free-tier quota spent: {exc}")
            return BackendUnavailable(f"gemini rate-limited (transient): {exc}")
        if exc.code in (401, 403):
            return BackendUnavailable(f"gemini rejected the API key: {exc}")
        if exc.code == 404:
            # Google returns NOT_FOUND when a configured model has been retired
            # or is not enabled for this account. The local model is a usable
            # fallback, unlike retrying this exact request into an HTTP 5xx.
            return BackendUnavailable(f"gemini model unavailable: {exc}")
        if exc.code >= 500:
            return BackendUnavailable(f"gemini server error: {exc}")
        # A 4xx that is not quota or auth is a bug in the request, and falling
        # back to local would hide it behind a working reply forever.
        return exc
    if isinstance(exc, (httpx.HTTPError, ConnectionError, OSError)):
        return BackendUnavailable(f"gemini unreachable: {exc}")
    return exc


class CloudBackend(ChatBackend):
    name = "cloud"
    # Gemini Flash-Lite does not need a 1.7B's deliberation crutch, and the
    # extra pass would spend a second free-tier request per turn to get it.
    supports_reasoning = False

    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float,
        max_tokens: int,
        timeout: float = 30.0,
        max_retries: int = 1,
    ) -> None:
        if not api_key:
            raise BackendUnavailable("GEMINI_API_KEY is not set")
        self.model = model
        self._is_gemini_3 = model.startswith("gemini-3")
        client_options: dict = {
            "model": model,
            "google_api_key": api_key,
            "max_output_tokens": max_tokens,
            "timeout": timeout,
            "max_retries": max_retries,
        }
        if self._is_gemini_3:
            # Gemini 3.x rejects the old numeric thinking budget and recommends
            # its default sampling values. "minimal" is the closest supported
            # low-latency setting to the old no-thought Flash-Lite path.
            client_options["thinking_level"] = "minimal"
        else:
            client_options.update({
                "temperature": temperature,
                "thinking_budget": 0,
                "include_thoughts": False,
            })
        self._client = ChatGoogleGenerativeAI(
            # One retry, not the library's default of six: this sits in front of
            # a person waiting to hear a sentence, and the fallback to local is
            # a better answer to a slow cloud than a longer wait.
            **client_options,
        )

    def _generation_options(self, options: GenerationOptions) -> dict:
        """Return only model-compatible per-call decoding controls."""
        if self._is_gemini_3:
            return {"max_output_tokens": options.max_tokens}
        return {
            "temperature": options.temperature,
            "max_output_tokens": options.max_tokens,
        }

    def prepare(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        """Persona + few-shot in front; the turn's own notes folded into the ask.

        The local path keeps its turn note as a system message sitting between
        the history and the question. That placement cannot survive here --
        LangChain hoists a SystemMessage into Gemini's `system_instruction`
        wherever it appears, which would move the note away from the question it
        describes and out of the persona's way. So a non-leading system message
        is folded into the user turn under a visible header instead.

        Folding was rejected for the local backend for a real, measured reason
        (a 1.7B answered the note instead of the question), which does not
        transfer: the note is labelled, and Flash-Lite reads a labelled context
        block as context.
        """
        head: list[dict[str, str]] = [{"role": "system", "content": load_system_prompt()}]
        head.extend(load_fewshot())

        notes: list[str] = []
        body: list[dict[str, str]] = []
        for message in messages:
            if message["role"] == "system":
                notes.append(message["content"])
            else:
                body.append(dict(message))

        if notes and body and body[-1]["role"] == "user":
            joined = "\n\n".join(notes)
            body[-1]["content"] = f"{_NOTE_HEADER}\n{joined}\n\n{body[-1]['content']}"
        elif notes:
            head.append({"role": "system", "content": "\n\n".join(notes)})

        return [*head, *body]

    @staticmethod
    def _read_usage(metadata: dict | None, usage: Usage) -> None:
        if not metadata:
            return
        usage.input_tokens = int(metadata.get("input_tokens") or 0)
        usage.output_tokens = int(metadata.get("output_tokens") or 0)

    async def agenerate(
        self, messages: list[dict[str, str]], options: GenerationOptions
    ) -> Generation:
        try:
            reply = await self._client.ainvoke(
                self.prepare(messages),
                **self._generation_options(options),
            )
        except Exception as exc:
            raise _translate(exc) from exc
        usage = Usage()
        self._read_usage(getattr(reply, "usage_metadata", None), usage)
        return Generation(text=_text_of(reply), usage=usage)

    async def astream(
        self,
        messages: list[dict[str, str]],
        options: GenerationOptions,
        usage: Usage,
    ) -> AsyncIterator[str]:
        try:
            async for chunk in self._client.astream(
                self.prepare(messages),
                **self._generation_options(options),
            ):
                self._read_usage(getattr(chunk, "usage_metadata", None), usage)
                if text := _text_of(chunk):
                    yield text
        except Exception as exc:
            raise _translate(exc) from exc


def _text_of(message) -> str:
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block if isinstance(block, str) else str(block.get("text", ""))
            for block in content
            if isinstance(block, (str, dict))
        )
    text = getattr(message, "text", None)
    if callable(text):
        return text()
    return str(content or text or "")
