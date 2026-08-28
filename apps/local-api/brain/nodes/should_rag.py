from __future__ import annotations

import re

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
    r"|i (already |have already )(said|told|mentioned|explained)"
    r"|say (it|that) again|repeat (that|it|yourself)"
    r"|lại nói|bạn vừa|tôi vừa|mày vừa|tao vừa"
    r"|nói lại|nhắc lại|gọi tôi là|gọi tao là"
    r"|(?:tôi|mình|tao) đã (?:nói|bảo|kể|nhắc|giải thích)"
    r"|đã (?:nói|bảo|kể|nhắc) (?:rồi|mà)"
    r")\b",
    re.IGNORECASE,
)

# The user handing a choice back to the assistant: "cứ chọn đi", "tùy bạn",
# "sao cũng được". This sits beside the decline detector because it is the same
# kind of signal -- a turn whose meaning is about the *previous* turn rather
# than about any topic -- but it has the opposite consequence. A decline means
# stop asking and let go; this one means stop asking and *deliver*.
#
# Two things follow from it, which is why it is a named predicate rather than a
# regex buried in one caller. It names no subject, so it must never reach the
# retriever: an embedding of "ok bạn cứ chọn" ranks memories by tone, not by
# subject, and returns a best-of-a-bad-lot hit. And it is an answer to a
# question the assistant already asked, so `generate` has to forbid asking it
# a second time -- the observed failure was exactly that loop, the assistant
# offering a choice, being told to choose, and offering the choice again.
#
# The addressed forms ("bạn chọn đi") are unambiguous wherever they appear, but
# only for verbs that cannot head a question: "bạn quyết định" is delegation
# while a bare "bạn nói" is not, so the speech verbs are admitted only with a
# following imperative particle ("bạn kể đi"). The unaddressed form ("cứ kể
# đi") is read as delegation only at the head of the utterance, otherwise
# "mình cứ nói thôi" would match it.
_DELEGATING_CHOICE_PATTERNS = re.compile(
    r"(?:"
    r"(?:bạn|cậu|mày|em)\s+(?:cứ\s+|tự\s+)?(?:chọn|quyết định|quyết)\b"
    r"|(?:bạn|cậu|mày|em)\s+(?:cứ\s+|tự\s+)?(?:kể|nói|chọn|làm)\s+(?:đi|nhé|thử|luôn)\b"
    r"|tùy\s+(?:bạn|cậu|mày|em|ý\s+bạn)\b"
    r"|(?:sao|gì|thế\s+nào|nào)\s+cũng\s+được\b"
    r"|^(?:ok(?:e|ay)?|ừ|ờ|vâng|dạ|thôi)?[\s,]*cứ\s+(?:chọn|kể|nói)\b"
    r"|\byou\s+(?:choose|decide|pick)\b|\byour\s+(?:choice|call)\b"
    r"|\bup\s+to\s+you\b|\bwhatever\s+you\s+(?:want|like|prefer)\b"
    r"|\beither\s+(?:one|way)\b"
    r")",
    re.IGNORECASE,
)

_DECLINING_ELABORATION_PATTERNS = re.compile(
    r"(?:"
    r"không (?:có|thấy có) (?:gì|cái gì) (?:đặc biệt|để kể)"
    r"|nothing (?:special|to (?:say|tell|share))"
    r")",
    re.IGNORECASE,
)

# Minimum word count before RAG is considered worthwhile.
_MIN_WORDS_FOR_RAG = 4


def is_delegating_choice(query: str) -> bool:
    """Detect the user handing the choice back: 'cứ chọn đi', 'tùy bạn'."""
    return bool(_DELEGATING_CHOICE_PATTERNS.search(query.strip()))


def is_declining_to_elaborate(query: str) -> bool:
    """Detect an explicit 'nothing to add' boundary, not a generic correction."""
    return bool(_DECLINING_ELABORATION_PATTERNS.search(query.strip()))


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

    # Delegation names no subject to retrieve on; see the pattern's own note.
    if is_delegating_choice(stripped):
        return False

    return True
