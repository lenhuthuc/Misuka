"""The slow path: a cloud vision-language model, called only when asked.

Nothing in this module runs during ingest. It fires when the router has
already established that neither the detector nor the OCR text can answer the
question -- so every call here is one the fast path could not avoid, and the
latency and cost are visible in `AnswerResult.source == "vlm"`.

The image sent is the buffered thumbnail, never the original upload.
"""
from __future__ import annotations

import asyncio
import base64
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

# The extracted data goes in ahead of the question so the model treats it as
# context rather than as an instruction, and the hedge about OCR errors is
# there because a confident misreading is the failure mode users notice.
VLM_PROMPT_TEMPLATE = (
    "Bạn nhận một ảnh và dữ liệu đã trích sẵn bằng OCR/detector (có thể sai sót nhỏ).\n"
    "Dữ liệu trích sẵn: {fast_summary}\n"
    "Câu hỏi: {question}\n"
    "Trả lời bằng tiếng Việt, ngắn gọn, chỉ dựa trên ảnh. "
    "Nếu ảnh không đủ thông tin, nói rõ."
)

DESCRIBE_QUESTION = "Mô tả ngắn gọn nội dung của ảnh này."

# Shown instead of an exception. The turn continues; the user gets a sentence
# they can act on rather than a traceback in the log and silence in the chat.
FRIENDLY_ERROR = (
    "Mình chưa xem kỹ được ảnh lúc này, bạn thử hỏi lại sau một chút nhé."
)


def build_prompt(question: str, fast_summary: str) -> str:
    return VLM_PROMPT_TEMPLATE.format(fast_summary=fast_summary, question=question)


class VLMClient(ABC):
    """One call: a question, the pre-extracted text, and a JPEG."""

    @abstractmethod
    async def ask(self, *, question: str, fast_summary: str, image_jpeg: bytes) -> str:
        """Return the answer text. Must not raise -- see FRIENDLY_ERROR."""

    async def aclose(self) -> None:
        return None


class OpenAICompatibleVLMClient(VLMClient):
    """Any provider speaking OpenAI `chat/completions` with image parts.

    Provider and model come from config, so switching between OpenAI, a local
    vLLM, OpenRouter or a Gemini-compatible gateway is a `.env` change.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: float = 15.0,
        max_retries: int = 1,
        max_tokens: int = 300,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._timeout = timeout_seconds
        self._max_retries = max_retries
        self._max_tokens = max_tokens
        self._client = None

    def _http(self):
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self._timeout,
                headers={"Authorization": f"Bearer {self._api_key}"} if self._api_key else {},
            )
        return self._client

    async def ask(self, *, question: str, fast_summary: str, image_jpeg: bytes) -> str:
        prompt = build_prompt(question, fast_summary)
        data_uri = "data:image/jpeg;base64," + base64.b64encode(image_jpeg).decode("ascii")
        payload = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }],
        }

        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = await self._http().post("/chat/completions", json=payload)
                response.raise_for_status()
                body = response.json()
                text = body["choices"][0]["message"]["content"]
                return (text or "").strip() or FRIENDLY_ERROR
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "vision.vlm attempt %d/%d failed: %s",
                    attempt + 1, self._max_retries + 1, exc,
                )
                if attempt < self._max_retries:
                    # Immediate retry would land inside the same provider blip.
                    await asyncio.sleep(0.5)

        logger.error("vision.vlm giving up after %d attempt(s): %s", self._max_retries + 1, last_error)
        return FRIENDLY_ERROR

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


class NullVLMClient(VLMClient):
    """Stand-in used when no provider is configured.

    Keeps the pipeline importable and the fast path fully working on a machine
    with no API key, instead of failing at construction time.
    """

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def ask(self, *, question: str, fast_summary: str, image_jpeg: bytes) -> str:
        self.calls.append({"question": question, "fast_summary": fast_summary})
        logger.info("vision.vlm not configured; returning the friendly fallback")
        return (
            "Mình chỉ đọc được những gì hiện rõ trong ảnh thôi, "
            "chưa phân tích sâu hơn được."
        )
