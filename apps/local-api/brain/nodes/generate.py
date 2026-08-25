from __future__ import annotations

# Block order is a latency decision, not a stylistic one. Ollama caches the
# longest matching prompt prefix, so a block placed before one that churns is
# re-prefilled along with it. These are ordered most-stable first: the rules are
# fixed, response policy changes per turn, and retrieved context changes
# whenever retrieval fires.
#
# The instructions are in English on purpose. The runner here is a 1.5B Qwen,
# and its instruction-following is measurably stronger in English than in
# Vietnamese — a fully Vietnamese system prompt made it answer "được" to an
# arithmetic question, and made the 3B answer in Chinese. English rules with an
# explicit Vietnamese *output* rule was the only combination that held.
#
# Rule 5 exists because the previous prompt was a retrieval-QA prompt ("use the
# provided context to answer... if the context does not contain enough
# information, say so honestly"). Against this app's actual context — whatever
# RAG retrieved — that instruction fires on every unrelated question: asked what
# one plus one is, the model reported that it could not determine the answer
# from what it had been given.
SYSTEM_PROMPT = """\
You are Mitsuka, a warm companion having a spoken conversation with the user.

Rules, in priority order:
1. ALWAYS reply in Vietnamese. Never reply in English or Chinese.
2. You are being read aloud. Keep it to 1-3 short sentences. No lists, no
   markdown, no emoji, no code blocks.
3. The user's words arrive from speech recognition, so they may be misheard,
   misspelled or missing a word. Answer what they most plausibly meant, and
   only ask them to repeat if you genuinely cannot guess.
4. For a factual, arithmetic or general-knowledge question, just answer it from
   your own knowledge. Answer first, elaborate only if it is short.
5. The notes below are optional background, NOT the source of your answers.
   Ignore anything in them unrelated to what was just said, and never tell the
   user that your notes lack the information.
6. Only discuss what you are or how you work if the user actually asked.
{response_policy_line}
Background notes:
{context}
"""


def _fit_history(recent: list[dict], char_budget: int) -> list[dict[str, str]]:
    """Keep the newest messages that fit the budget, dropping oldest first.

    A message *count* does not bound a prompt: ten replies of three sentences
    and ten replies of three paragraphs differ by an order of magnitude. Since
    prefill cost is linear in prompt length and every turn re-sends the whole
    window, an unbounded history makes each turn slower than the last.
    """
    kept: list[dict[str, str]] = []
    spent = 0
    for row in reversed(recent):
        cost = len(row["content"])
        if kept and spent + cost > char_budget:
            break
        kept.append({"role": row["role"], "content": row["content"]})
        spent += cost

    kept.reverse()
    return kept


def build_messages(
    query: str,
    context: str,
    recent: list[dict],
    history_char_budget: int = 3000,
    response_policy_instruction: str = "",
) -> list[dict[str, str]]:
    """Build the full message list for a chat completion call.

    Takes the already-fetched recent window rather than the memory service:
    the caller needs those same rows to tell the retriever which turns the
    prompt already covers, and fetching them twice per turn served nobody.

    Both variable-length parts carry their own budget — history and retrieved
    context — because prefill cost is linear in prompt length and the two grow
    on different schedules.

    Stored output emotion remains metadata for memory and UI. It is not fed
    back as a fixed mood prompt; the current user's VAD produces the small
    behavioural policy passed in by the turn orchestrator.
    """
    history = _fit_history(recent, history_char_budget)
    response_policy_line = f"{response_policy_instruction}\n" if response_policy_instruction else ""
    system = SYSTEM_PROMPT.format(
        context=context, response_policy_line=response_policy_line,
    )
    return [{"role": "system", "content": system}] + history + [{"role": "user", "content": query}]
