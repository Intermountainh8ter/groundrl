"""The prompt a model sees, shared by SFT, GRPO and evaluation.

One function builds the chat messages for an :class:`~groundrl.example.Example`, so a prompt
cannot drift between training and evaluation. Chat templating itself is left to the model's own
tokenizer (``tokenizer.apply_chat_template``); :func:`render_chat_stub` is a dependency-free
ChatML rendering used only for dry runs and token estimates.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

from .example import Example
from .protocol import ABSTAIN_TEXT

PROMPT_VERSION = 1

SYSTEM_PROMPT = (
    "You answer questions about Ethereum Improvement Proposals (EIPs) using only the context "
    "chunks provided by the user.\n"
    "Rules:\n"
    "1. Write short sentences. End every sentence with a citation naming the chunk it relies on, "
    "in exactly this form: [eip-N#CCC], for example [eip-1559#011].\n"
    "2. Cite only chunk ids that appear in the context. A sentence may cite more than one chunk.\n"
    "3. Quote numbers, hex values and identifiers exactly as they appear in the context.\n"
    "4. Answer only what was asked. Do not add other facts or alternatives.\n"
    f"5. If the context does not contain the answer, reply with exactly: {ABSTAIN_TEXT}"
)


def format_context(context: Mapping[str, str]) -> str:
    """Chunks as ``[id]`` header lines followed by their text, separated by blank lines."""
    return "\n\n".join(f"[{cid}]\n{text}" for cid, text in context.items())


def user_message(ex: Example) -> str:
    return f"Context:\n\n{format_context(ex.context)}\n\nQuestion: {ex.task.question}"


def build_messages(ex: Example) -> list[dict[str, str]]:
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message(ex)}]


def render_chat_stub(messages: Sequence[Mapping[str, str]], add_generation_prompt: bool = True,
                     enable_thinking: bool = False) -> str:
    """ChatML rendering in the style of the Qwen3 template (for token estimates only).

    The real template of the real tokenizer is always used for training and evaluation; this
    function exists so that dry runs need no tokenizer download.
    """
    out = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
    if add_generation_prompt:
        out += "<|im_start|>assistant\n"
        if not enable_thinking:
            out += "<think>\n\n</think>\n\n"
    return out
