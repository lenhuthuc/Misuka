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
#
# The adverb between the pronoun and the verb is a *slot*, not a fixed choice.
# It used to be spelled `(?:cứ\s+|tự\s+)?`, which accepts exactly two words and
# silently rejected every other way of saying the same thing -- "bạn tự động
# chọn", "bạn cứ tự nhiên chọn", "bạn tuỳ ý chọn". That mattered far more than
# a missed regex usually does: this one predicate gates the deliver-now
# instruction, the trailing-question suppressor and RAG suppression all at
# once, so a phrasing that slips through here turns the entire anti-loop path
# off and the assistant goes back to offering the choice it was just told to
# make. Longest alternatives first so "tự động" is not eaten by "tự".
_DELEGATION_ADVERB = (
    r"(?:tự\s*động|tự\s+nhiên|tuỳ\s*ý|tùy\s*ý|chủ\s+động|thoải\s+mái|cứ|tự)"
)
_DELEGATION_PRONOUN = r"(?:bạn|cậu|mày|em)"

_DELEGATING_CHOICE_PATTERNS = re.compile(
    r"(?:"
    # "bạn chọn", "bạn tự động chọn", "bạn cứ tự nhiên quyết định". The
    # lookahead is what keeps "bạn chọn cái gì vậy" out: an interrogative right
    # after the verb makes it the user asking, not the user deferring.
    # The "định" is consumed atomically. Without that, "bạn quyết định gì rồi"
    # fails the lookahead on "quyết định", backtracks to the shorter "quyết",
    # and then passes -- because what follows *that* is "định", not a question
    # word.
    rf"{_DELEGATION_PRONOUN}(?:\s+{_DELEGATION_ADVERB})*\s+(?:chọn|quyết(?>(?:\s*định)?))\b"
    r"(?!\s+(?:cái\s+)?(?:gì|nào|ai|đâu|sao)\b)"
    # The imperative particle may sit a few words after the verb: "bạn cứ nói
    # về nó đi", "bạn cụ thể cho mình về chủ đề đó đi". The span between is
    # bounded and may not contain an interrogative, so "bạn nói xem cái nào
    # tốt hơn đi" is left alone and "bạn nói gì thế" -- no particle at all --
    # stays a question rather than a delegation.
    rf"|{_DELEGATION_PRONOUN}(?:\s+{_DELEGATION_ADVERB})*\s+(?:kể|nói|chọn|làm|cụ\s+thể)"
    r"(?:\s+(?!gì\b|nào\b|sao\b|ai\b|đâu\b)\S+){0,6}?\s+(?:đi|nhé|thử|luôn)\b"
    # Both tone placements: "tùy" and "tuỳ" are the same word, and the mark
    # sits on a different vowel in each, so a character class does not cover it.
    r"|t(?:ùy|uỳ)\s+(?:bạn|cậu|mày|em|ý\s+bạn)\b"
    r"|(?:sao|gì|thế\s+nào|nào)\s+cũng\s+được\b"
    # "thích cái nào thì chọn" -- the choice handed back as a conditional.
    r"|(?:thích|muốn)\s+(?:cái\s+)?(?:nào|gì)\s+thì\s+(?:chọn|lấy|làm|kể|nói)\b"
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
