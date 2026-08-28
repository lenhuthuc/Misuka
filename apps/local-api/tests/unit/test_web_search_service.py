"""WebSearchService: the same degrade-on-failure contract RAGService gives
retrieval -- a search failure must never fail the turn, only empty its context.
"""
from __future__ import annotations

import pytest

from brain.web_search_service import WebSearchService


class StubDDGS:
    """Stands in for `ddgs.DDGS` -- records the call, returns canned hits."""

    calls: list[dict] = []
    results: list[dict] = []
    error: Exception | None = None

    def __init__(self, timeout: float) -> None:
        self.timeout = timeout

    def text(self, query: str, region: str, max_results: int) -> list[dict]:
        StubDDGS.calls.append({"query": query, "region": region, "max_results": max_results})
        if StubDDGS.error is not None:
            raise StubDDGS.error
        return StubDDGS.results


@pytest.fixture(autouse=True)
def _reset_stub():
    StubDDGS.calls = []
    StubDDGS.results = []
    StubDDGS.error = None
    yield


@pytest.fixture
def service(monkeypatch) -> WebSearchService:
    monkeypatch.setattr("brain.web_search_service.DDGS", StubDDGS)
    return WebSearchService(max_results=3, context_char_budget=200, region="vn-vi")


async def test_results_are_formatted_with_title_body_and_source(service):
    StubDDGS.results = [
        {"title": "Thời tiết Hà Nội", "body": "Nắng, 32 độ.", "href": "https://example.com/weather"},
    ]

    results, context = await service.build_context("thời tiết hà nội hôm nay")

    assert results == StubDDGS.results
    assert "Thời tiết Hà Nội" in context
    assert "Nắng, 32 độ." in context
    assert "https://example.com/weather" in context


async def test_no_results_returns_empty_context(service):
    StubDDGS.results = []

    results, context = await service.build_context("query với không kết quả")

    assert results == []
    assert context == ""


async def test_search_failure_degrades_to_empty_context_not_an_exception(service):
    from ddgs.exceptions import DDGSException

    StubDDGS.error = DDGSException("engine unavailable")

    results, context = await service.build_context("bất kỳ câu hỏi nào")

    assert results == []
    assert context == ""


async def test_context_is_trimmed_to_the_char_budget(monkeypatch):
    monkeypatch.setattr("brain.web_search_service.DDGS", StubDDGS)
    service = WebSearchService(max_results=5, context_char_budget=50, region="vn-wt")
    StubDDGS.results = [
        {"title": "A" * 40, "body": "first result body text", "href": "https://a.example"},
        {"title": "B" * 40, "body": "second result body text", "href": "https://b.example"},
    ]

    _, context = await service.build_context("query dài")

    assert "A" * 40 in context
    assert "B" * 40 not in context


async def test_query_is_forwarded_with_region_and_max_results(service):
    StubDDGS.results = []

    await service.build_context("truy vấn kiểm tra")

    assert StubDDGS.calls == [{"query": "truy vấn kiểm tra", "region": "vn-vi", "max_results": 3}]
