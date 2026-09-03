"""The one place a prompt is read from disk, for both backends.

Mitsuka's persona lives in `assets/models/LLM/Modelfile` and is baked into
`mitsuka-ft` by `ollama create`. That is not a stylistic choice and it
constrains everything here: Ollama injects the Modelfile SYSTEM only when the
caller's first message is not itself a system message, so the local backend
must NOT send a system prompt (see the layout rule at the top of
`brain/nodes/generate.py` -- sending one silently replaces the persona, and the
model stops calling itself Mitsuka). Gemini has no Modelfile, so the cloud
backend must send one.

Same text, two delivery mechanisms. `system_prompt.txt` is therefore a verbatim
copy of the Modelfile's SYSTEM block, and `check_persona_drift` is what keeps
"verbatim" true -- a copy nobody verifies is how the two backends start
speaking in different voices three months from now, which is the exact failure
the single-prompt rule exists to prevent.
"""
from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"
_SYSTEM_PROMPT_PATH = _PROMPTS_DIR / "system_prompt.txt"
_FEWSHOT_PATH = _PROMPTS_DIR / "fewshot.json"
_MODELFILE_PATH = (
    Path(__file__).resolve().parents[4] / "assets" / "models" / "LLM" / "Modelfile"
)

# SYSTEM """...""" in a Modelfile. Ollama also accepts a bare single-line form,
# hence the alternation; the triple-quoted branch is what this repo uses.
_SYSTEM_BLOCK = re.compile(
    r'^\s*SYSTEM\s+(?:"""(?P<block>.*?)"""|(?P<line>.+))\s*$',
    re.MULTILINE | re.DOTALL | re.IGNORECASE,
)


def parse_modelfile_system(modelfile_text: str) -> str:
    """Extract the SYSTEM block from a Modelfile, or "" if it has none."""
    match = _SYSTEM_BLOCK.search(modelfile_text)
    if match is None:
        return ""
    return (match.group("block") or match.group("line") or "").strip()


@lru_cache(maxsize=1)
def load_system_prompt() -> str:
    """The persona text. Sent to cloud; carried by the weights on local."""
    return _SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()


@lru_cache(maxsize=1)
def load_fewshot() -> list[dict[str, str]]:
    """Few-shot exchanges, as chat messages. Cloud only -- see `backends.local`.

    Empty list when the file is absent: few-shot is an aid for a model that has
    not been fine-tuned on this voice, not a correctness requirement.
    """
    if not _FEWSHOT_PATH.exists():
        return []
    raw = json.loads(_FEWSHOT_PATH.read_text(encoding="utf-8"))
    messages: list[dict[str, str]] = []
    for pair in raw:
        messages.append({"role": "user", "content": pair["user"]})
        messages.append({"role": "assistant", "content": pair["assistant"]})
    return messages


def check_persona_drift() -> str | None:
    """Compare `system_prompt.txt` against the Modelfile SYSTEM block.

    Returns a human-readable description of the difference, or None when they
    match (or when the Modelfile is not on this machine -- a deploy that ships
    only `apps/local-api/` is not a drift, it is a missing file, and it must not
    fail startup).
    """
    if not _MODELFILE_PATH.exists():
        return None
    baked = parse_modelfile_system(_MODELFILE_PATH.read_text(encoding="utf-8"))
    if not baked:
        return f"{_MODELFILE_PATH} has no SYSTEM block to compare against"
    if baked == load_system_prompt():
        return None
    return (
        f"persona drift: {_SYSTEM_PROMPT_PATH.name} no longer matches the SYSTEM "
        f"block in {_MODELFILE_PATH}. The cloud backend sends the former and the "
        f"local backend is trained on the latter, so the two now speak "
        f"differently. Re-copy one onto the other and re-run `ollama create`."
    )


def warn_on_persona_drift() -> None:
    """Log the drift check at startup. Never raises."""
    try:
        drift = check_persona_drift()
    except OSError:
        logger.exception("persona drift check failed to read its inputs")
        return
    if drift:
        logger.warning("%s", drift)
