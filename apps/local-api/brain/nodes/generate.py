from __future__ import annotations

# Block order is a latency decision, not a stylistic one. Ollama caches the
# longest matching prompt prefix, so a block placed before one that churns is
# re-prefilled along with it. These are ordered most-stable first: facts change
# only when the curator learns something, response policy changes per turn, and
# retrieved context changes whenever retrieval fires.
#
# Measured on this setup with the same 606-character facts block: placing it
# before a changing per-turn block avoids re-prefilling the facts every turn.
SYSTEM_PROMPT = """\
You are BrainMaster, a helpful and knowledgeable assistant.
Use the provided context to answer the user's question accurately and concisely.
If the context does not contain enough information, say so honestly.
{facts_block}{response_policy_line}
Context:
{context}
"""


def _fit_facts(facts: list[dict], char_budget: int) -> str:
    """Render stored facts as a prompt block, newest first, within a budget.

    Facts are the only part of the prompt that persists across sessions, so
    they are worth their prefill cost — but the table only grows, and every
    turn re-sends the whole block. Newest-first means a table that has outgrown
    the budget keeps what the curator learned most recently.
    """
    lines: list[str] = []
    spent = 0
    for fact in facts:
        line = f"- {fact['key']}: {fact['value']}"
        if lines and spent + len(line) > char_budget:
            break
        lines.append(line)
        spent += len(line)

    if not lines:
        return ""
    return "What you know about the user:\n" + "\n".join(lines) + "\n"


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
    facts: list[dict] | None = None,
    facts_char_budget: int = 600,
    response_policy_instruction: str = "",
) -> list[dict[str, str]]:
    """Build the full message list for a chat completion call.

    Takes the already-fetched recent window rather than the memory service:
    the caller needs those same rows to tell the retriever which turns the
    prompt already covers, and fetching them twice per turn served nobody.

    Every variable-length part carries its own budget — history, retrieved
    context, and facts — because prefill cost is linear in prompt length and
    each of these grows on a different schedule.

    Stored output emotion remains metadata for memory and UI. It is not fed
    back as a fixed mood prompt; the current user's VAD produces the small
    behavioural policy passed in by the turn orchestrator.
    """
    history = _fit_history(recent, history_char_budget)
    facts_block = _fit_facts(facts or [], facts_char_budget)
    response_policy_line = f"{response_policy_instruction}\n" if response_policy_instruction else ""
    system = SYSTEM_PROMPT.format(
        context=context, response_policy_line=response_policy_line, facts_block=facts_block,
    )
    return [{"role": "system", "content": system}] + history + [{"role": "user", "content": query}]
