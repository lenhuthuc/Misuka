from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from brain.nodes.should_rag import is_delegating_choice

# Topics whose true answer changes over time -- neither the local model's
# frozen weights nor personal-conversation RAG can hold today's version of
# them. Deliberately narrow: this is the gate that decides whether to pay for
# a network call, so it favours precision (a real topic keyword) over recall
# (a bare time marker like "hôm nay" alone names no topic to search for).
_EXTERNAL_INFO_PATTERNS = re.compile(
    r"("
    r"thời tiết|dự báo thời tiết|nhiệt độ (?:hôm nay|ngày mai|bây giờ)"
    r"|tin tức|tin mới nhất|thời sự"
    r"|giá (?:vàng|xăng|dầu|cổ phiếu|bitcoin|đô la|usd)|tỷ giá"
    r"|kết quả (?:trận|bóng đá|xổ số)|xổ số|tỷ số"
    r"|weather (?:today|forecast|tomorrow)|latest news|breaking news"
    r"|stock price|exchange rate|gold price|oil price|lottery result"
    r")",
    re.IGNORECASE,
)

# Recommendation/comparison-seeking phrasing: "sản phẩm nào tốt", "nên mua
# gì" name no fixed topic the way "thời tiết" does, so they can't join the
# list above -- but the *phrasing itself* is the signal. Answering these well
# needs current listings/reviews, not the model's frozen weights, so this is
# still the precision-first "a real marker matched" gate, just matching a
# sentence pattern instead of a topic word.
_RECOMMENDATION_PATTERNS = re.compile(
    r"("
    r"(?:nào|gì) (?:tốt|hay|ngon|đáng mua|đáng dùng|đáng chọn)\b"
    r"|tốt nhất|hay nhất|ngon nhất|đáng mua nhất|đáng dùng nhất"
    r"|nên mua|nên dùng|nên chọn"
    r"|so sánh .+(?: với | và ).+"
    r"|which .+ is better|best .+ to buy|worth buying|worth it\b"
    r")",
    re.IGNORECASE,
)

# The third class, and the reason this file grew past a single predicate.
#
# The two gates above cover "what is true right now" and "what should I buy".
# Between them and RAG's "what did we already talk about" sits a hole that no
# gate owned: an ordinary question about the world that is neither live nor
# personal -- "tôm biển là con gì", "kể cho mình nghe về sao Hoả". A 1.7B Q4
# fine-tune has no reliable long-tail knowledge to answer those from, and with
# nothing retrieved and nothing searched it answers from nothing at all. The
# observed failure was an invented species ("tôm biển mọc lông") stated with
# the same confidence as a fact, then contradicted two turns later.
#
# This is the gate that decides whether such a turn gets grounded. It is a
# *narrow* addition on purpose -- the question this whole change had to answer
# was "surely not every turn hits the network", and the answer is that
# chit-chat, feelings, and anything about the two people talking still match
# nothing here.
_KNOWLEDGE_QUESTION_PATTERNS = re.compile(
    r"(?:"
    r"\blà\s+(?:con\s+|cái\s+|loài\s+|chất\s+|nước\s+|gì\b|ai\b)"
    r"|\bnghĩa\s+là\s+gì\b"
    r"|\b(?:tại\s+sao|vì\s+sao|do\s+đâu)\b"
    r"|\bkể\s+(?:cho\s+\S+\s+nghe\s+)?(?:về|chuyện)\b"
    r"|\b(?:cho\s+\S+\s+biết|nói\s+thêm)\s+về\b"
    r"|\bgiải\s+thích\b|\btìm\s+hiểu\s+về\b|\bthông\s+tin\s+về\b"
    r"|\bcó\s+(?:mấy|bao\s+nhiêu)\s+loại\b"
    r"|\b(?:hoạt\s+động|sinh\s+sống|sinh\s+sản|săn\s+mồi|xảy\s+ra|diễn\s+ra)"
    r"\s+(?:như\s+thế\s+nào|ra\s+sao|thế\s+nào)\b"
    r"|\bwhat\s+(?:is|are|was|were)\b|\bwho\s+(?:is|are|was)\b"
    r"|\bwhy\s+(?:is|are|do|does|did)\b|\bhow\s+(?:do|does|did)\b"
    r"|\btell\s+me\s+about\b|\bexplain\b"
    r")",
    re.IGNORECASE,
)

# The guard that keeps the gate above from swallowing RAG's territory. A
# question can be phrased exactly like a knowledge question and still be about
# the two people in the conversation -- "con mèo nhà mình tên gì", "bạn tên
# gì". Nothing on the open web knows those, so a search would spend a network
# call to return noise, and worse, put stranger's text in front of a model
# being asked about the user's own life. Checked before the knowledge gate,
# never before the live one: "giá vàng" stays a live query no matter who asks.
_PERSONAL_SCOPE_PATTERNS = re.compile(
    r"(?:"
    r"nhà\s+(?:mình|tôi|tớ|em)\b|của\s+(?:mình|tôi|tớ|chúng\s+mình|chúng\s+ta)\b"
    r"|(?:mình|tôi|tớ)\s+(?:là|có|thích|ghét|từng|đã|đang|vừa|sẽ)\b"
    r"|\bbạn\s+(?:tên|là\s+ai|thích|ghét|nhớ|có\s+nhớ|nghĩ|thấy)\b"
    r"|chúng\s+(?:ta|mình)\b"
    r"|\bmy\s+|\bour\s+|\byour\s+name\b|\bdo\s+you\s+remember\b"
    r")",
    re.IGNORECASE,
)

# DuckDuckGo is given a topic, not a sentence. "kể cho mình nghe về tôm biển"
# as a literal query ranks blog posts that contain that whole phrasing; "tôm
# biển" ranks the subject. Stripping the imperative lead-in and the spoken
# tail particles is the difference between a useful snippet and three
# unrelated ones -- and unrelated snippets are worse than none, because they
# arrive in the prompt under a header telling the model they are real.
_SEARCH_LEAD_IN = re.compile(
    r"^\s*(?:"
    r"kể\s+(?:cho\s+\S+\s+nghe\s+)?(?:về|chuyện)"
    r"|(?:cho\s+\S+\s+biết|nói\s+thêm)\s+về"
    r"|giải\s+thích(?:\s+về)?|tìm\s+hiểu\s+về|thông\s+tin\s+về"
    r"|tell\s+me\s+about|explain"
    r")\s+",
    re.IGNORECASE,
)

_SEARCH_TAIL = re.compile(
    r"(?:[\s,]+(?:nhé|nhỉ|đi|với|ạ|thế|vậy|hả|hử|nha|nào))*\s*[?.!]*\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class WebSearchDecision:
    """Whether this turn needs the web, what to search for, and why.

    `kind` is carried rather than a bare bool because the caller does two
    different things with it: "live" and "knowledge" both fetch, but only
    "knowledge" retunes decoding (see `brain.response_policy`) -- a grounded
    factual answer wants a colder, longer reply than a chatty one.
    """

    kind: str = ""            # "" | "live" | "knowledge"
    query: str = ""
    inherited: bool = False

    @property
    def should_search(self) -> bool:
        return bool(self.kind and self.query)


def to_search_query(text: str) -> str:
    """Reduce a spoken question to the topic terms worth sending to a search engine."""
    stripped = _SEARCH_LEAD_IN.sub("", text.strip())
    return _SEARCH_TAIL.sub("", stripped).strip()


def is_knowledge_question(query: str) -> bool:
    """Return True for an open-world question the local weights cannot be trusted on."""
    stripped = query.strip()
    if _PERSONAL_SCOPE_PATTERNS.search(stripped):
        return False
    return bool(_KNOWLEDGE_QUESTION_PATTERNS.search(stripped))


def should_use_web_search(query: str) -> bool:
    """Return True if the query is asking about live, external information.

    This is independent of `should_use_rag`: RAG answers "what did we already
    talk about", this answers "what is true about the world right now" --
    they are not competing for the same turn, so this is not gated on RAG's
    own eligibility check.
    """
    stripped = query.strip()
    return bool(_EXTERNAL_INFO_PATTERNS.search(stripped) or _RECOMMENDATION_PATTERNS.search(stripped))


def _classify(text: str) -> str:
    if should_use_web_search(text):
        return "live"
    return "knowledge" if is_knowledge_question(text) else ""


def decide_web_search(
    query: str,
    recent: Sequence[dict] = (),
    *,
    knowledge_enabled: bool = True,
) -> WebSearchDecision:
    """Decide this turn's web grounding, inheriting the topic when the turn has none.

    The current utterance is classified first. If it carries no topic of its
    own *and* is the user handing the choice back ("ok bạn cứ chọn"), the need
    is inherited from the newest substantive user turn instead -- because that
    is where the subject still lives. Without this the loop in the observed
    failure could not be broken by grounding at all: the turn that most needed
    a real fact was the one whose text was five topic-free words, and searching
    "ok bạn cứ chọn" returns nothing worth putting in a prompt.

    Only the *newest* substantive user turn is consulted, not the whole window.
    Scanning further back would resurrect an abandoned subject on any turn that
    happened to look like delegation.
    """
    kind = _classify(query)
    if kind == "knowledge" and not knowledge_enabled:
        kind = ""
    if kind:
        return WebSearchDecision(kind=kind, query=to_search_query(query))

    if not is_delegating_choice(query):
        return WebSearchDecision()

    for row in reversed(list(recent)):
        if row.get("role") != "user":
            continue
        content = str(row.get("content", ""))
        if is_delegating_choice(content):
            continue
        inherited_kind = _classify(content)
        if inherited_kind == "knowledge" and not knowledge_enabled:
            return WebSearchDecision()
        if inherited_kind:
            return WebSearchDecision(
                kind=inherited_kind, query=to_search_query(content), inherited=True,
            )
        return WebSearchDecision()

    return WebSearchDecision()
