from __future__ import annotations

import asyncio
import logging

from ddgs import DDGS
from ddgs.exceptions import DDGSException

logger = logging.getLogger(__name__)


class WebSearchService:
    """Thin async wrapper over DDGS (DuckDuckGo) text search.

    DDGS is a synchronous client that does its own blocking HTTP, so every
    call runs on a worker thread via `asyncio.to_thread` rather than blocking
    the event loop other turns and background tasks share.
    """

    def __init__(
        self,
        max_results: int = 3,
        context_char_budget: int = 800,
        timeout_seconds: float = 6.0,
        region: str = "vn-vi",
    ) -> None:
        self._max_results = max_results
        self._context_char_budget = context_char_budget
        self._timeout_seconds = timeout_seconds
        self._region = region

    async def build_context(self, query: str) -> tuple[list[dict], str]:
        """Search the web and format the results as prompt context.

        Returns `([], "")` on any search failure -- a network hiccup or an
        engine outage degrades the turn to no web context rather than failing
        it, the same contract `RAGService` gives the retrieval path.
        """
        try:
            results = await asyncio.to_thread(self._search_sync, query)
        except DDGSException as exc:
            logger.warning("web_search | query failed: %s", exc)
            return [], ""
        return results, self._format_context(results)

    def _search_sync(self, query: str) -> list[dict]:
        return DDGS(timeout=self._timeout_seconds).text(
            query,
            region=self._region,
            max_results=self._max_results,
        )

    def _format_context(self, results: list[dict]) -> str:
        if not results:
            return ""
        lines: list[str] = []
        spent = 0
        for r in results:
            title = (r.get("title") or "").strip()
            body = (r.get("body") or "").strip()
            url = (r.get("href") or "").strip()
            line = f"- {title}: {body} (nguồn: {url})"
            if lines and spent + len(line) > self._context_char_budget:
                break
            lines.append(line)
            spent += len(line)
        return "\n".join(lines)
