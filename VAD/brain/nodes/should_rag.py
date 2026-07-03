from __future__ import annotations

import re

from brain.state import BrainState

# Exact-match conversational phrases (after stripping punctuation).
_CONVERSATIONAL_PATTERNS = re.compile(
    r"^("
    r"hi|hello|hey|chào|xin chào|alo"
    r"|bye|goodbye|bye bye|see you|tạm biệt|hẹn gặp lại"
    r"|thanks|thank you|thank you very much|cảm ơn|cảm ơn bạn"
    r"|ok|okay|oke|alright|sure|got it|understood|i see|i know"
    r"|yes|no|yep|nope|yeah|nah|có|không|vâng|dạ"
    r"|good morning|good afternoon|good evening|good night|chúc ngủ ngon"
    r"|how are you|are you ok|you good|bạn khỏe không"
    r"|nice|cool|great|awesome|wow|interesting"
    r")$",
    re.IGNORECASE,
)

# Queries that refer to the current conversation context — RAG can never help these.
# "what did you say", "what name did you call me", "say that again", etc.
_SELF_REFERENCE_PATTERNS = re.compile(
    r"\b("
    r"you (just |did |said?|called?|mentioned?|told?|asked?|answered?)"
    r"|what (did you|have you|name did you|did i)"
    r"|i (just |did |said?|told?|asked?)"
    r"|say (it|that) again|repeat (that|it|yourself)"
    r"|lại nói|bạn vừa|tôi vừa|mày vừa|tao vừa"
    r"|nói lại|nhắc lại|gọi tôi là|gọi tao là"
    r")\b",
    re.IGNORECASE,
)

# Minimum word count before RAG is considered worthwhile.
_MIN_WORDS_FOR_RAG = 4


def should_use_rag(query: str) -> bool:
    """Return True if the query is likely to benefit from vector retrieval."""
    stripped = query.strip().rstrip("?.!,;:")

    # Very short queries are almost always conversational
    if len(stripped.split()) < _MIN_WORDS_FOR_RAG:
        return False

    # Exact conversational phrase
    if _CONVERSATIONAL_PATTERNS.match(stripped):
        return False

    # Query refers to this conversation's own context — RAG has nothing useful
    if _SELF_REFERENCE_PATTERNS.search(stripped):
        return False

    return True


def make_should_rag_node():
    async def should_rag_node(state: BrainState) -> dict:
        use_rag = should_use_rag(state["query"])
        return {"use_rag": use_rag}

    return should_rag_node
