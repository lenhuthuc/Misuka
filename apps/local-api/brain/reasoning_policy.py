"""Adaptive, zero-I/O policy for selectively enabling model thinking.

Thinking is a fallback for substantive turns that neither recent conversation
nor long-term retrieval can explain. It is intentionally not a replacement for
retrieval: strong context keeps the fast non-thinking path.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from math import floor, log

from brain.nodes.should_rag import should_use_rag

_WORD = re.compile(r"\w+", re.UNICODE)
_STOP_WORDS = {
    "a", "ạ", "à", "bạn", "bị", "các", "cái", "cho", "có", "của", "đã",
    "đang", "để", "đi", "đó", "được", "gì", "hay", "không", "là", "lại",
    "mà", "mình", "một", "này", "nó", "nhé", "nhỉ", "những", "ơi", "rồi",
    "sao", "thế", "thì", "tôi", "trong", "và", "về", "với",
}


@dataclass(frozen=True)
class ReasoningPolicy:
    """Deliberation decision and the hidden pass's hard token budget."""

    enabled: bool = False
    token_budget: int = 0
    confidence: float = 0.0
    rag_score: float = 0.0
    history_score: float = 0.0


_REASONING_HEADER = (
    "Phân tích nội bộ cho câu hỏi kế tiếp. Chỉ dùng làm gợi ý; tự kiểm tra lại, "
    "không nhắc đến ghi chú này và không tiết lộ chuỗi suy luận:"
)


def inject_reasoning_note(
    messages: list[dict[str, str]],
    reasoning: str,
) -> list[dict[str, str]]:
    """Place hidden deliberation before the final query without losing persona.

    On a fresh conversation a system message at index zero suppresses Ollama's
    Modelfile SYSTEM, and an artificial leading assistant turn destabilises the
    trained Vietnamese persona. In that one case the directly relevant analysis
    is labelled inside the sole user turn. Once history exists, a late system
    note is safe and has stronger precedence.
    """
    if not reasoning.strip() or not messages:
        return list(messages)

    note = f"{_REASONING_HEADER}\n{reasoning.strip()}"
    enriched = list(messages)
    if len(enriched) == 1:
        query = enriched[0]["content"]
        enriched[0] = {
            "role": "user",
            "content": f"{note}\n\nLời người dùng hiện tại:\n{query}",
        }
    else:
        enriched.insert(len(enriched) - 1, {"role": "system", "content": note})
    return enriched


def _terms(text: str) -> set[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return {word for word in _WORD.findall(normalized) if len(word) > 1 and word not in _STOP_WORDS}


def history_relevance(query: str, recent: list[dict]) -> float:
    """Cheap lexical relevance for recent rows not yet present in the vector index.

    Each completed exchange is embedded only after speech finishes, so the
    vector-store evidence can lag the verbatim SQLite history. Token containment
    closes that gap without another embedding call on the latency-critical path.
    """
    query_terms = _terms(query)
    if not query_terms:
        return 0.0

    best = 0.0
    for row in recent:
        history_terms = _terms(str(row.get("content", "")))
        if not history_terms:
            continue
        best = max(best, len(query_terms & history_terms) / len(query_terms))
    return best


def derive_reasoning_policy(
    query: str,
    recent: list[dict],
    *,
    rag_score: float = 0.0,
    indexed_history_score: float = 0.0,
    activation_threshold: float = 0.50,
    min_tokens: int = 64,
    max_tokens: int = 192,
    token_scale: float = 0.65,
    enabled: bool = True,
) -> ReasoningPolicy:
    """Enable thinking when a substantive turn has weak supporting context.

    The budget follows a normalized negative-log curve as the strongest
    available context score falls from ``activation_threshold`` to zero. It is
    the hard ``num_predict`` cap of a separate hidden deliberation pass, so it
    cannot consume the answer's own generation budget.
    """
    lexical_history_score = history_relevance(query, recent)
    history_score = max(indexed_history_score, lexical_history_score)
    confidence = max(rag_score, history_score)

    bounded_scale = min(1.0, max(0.0, token_scale))
    if not enabled or bounded_scale <= 0.0 or not should_use_rag(query) or confidence >= activation_threshold:
        return ReasoningPolicy(
            confidence=confidence,
            rag_score=rag_score,
            history_score=history_score,
        )

    threshold = max(activation_threshold, 1e-6)
    bounded_min = max(0, min(min_tokens, max_tokens))
    bounded_max = max(bounded_min, max_tokens)
    clamped_confidence = min(threshold, max(0.0, confidence))

    # -log(score / threshold) has the desired shape but diverges at score=0.
    # A 1%-of-threshold epsilon makes the curve finite, then normalization maps
    # score=0 -> 1 and score=threshold -> 0 exactly. The logarithmic decay drops
    # the allowance quickly as evidence improves, while weak evidence retains a
    # meaningfully larger budget. Token counts always round toward the lower
    # bound so the policy never exceeds the continuous curve.
    epsilon = threshold * 0.01
    log_max = -log(epsilon / (threshold + epsilon))
    missing = -log((clamped_confidence + epsilon) / (threshold + epsilon)) / log_max
    missing = min(1.0, max(0.0, missing))
    continuous_budget = bounded_min + (bounded_max - bounded_min) * missing
    budget = floor(continuous_budget * bounded_scale)
    return ReasoningPolicy(
        enabled=True,
        token_budget=budget,
        confidence=confidence,
        rag_score=rag_score,
        history_score=history_score,
    )
