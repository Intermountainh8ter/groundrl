from __future__ import annotations

from groundrl.example import Example
from groundrl.prompts import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    build_messages,
    format_context,
    render_chat_stub,
    user_message,
)
from groundrl.protocol import ABSTAIN_TEXT
from tests.test_rewards import comparison, ex, multi, unanswerable


def test_messages_have_a_system_rule_set_and_a_user_turn():
    msgs = build_messages(ex())
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[0]["content"] == SYSTEM_PROMPT


def test_system_prompt_states_the_protocol_the_verifier_enforces():
    assert "[eip-N#CCC]" in SYSTEM_PROMPT and "[eip-1559#011]" in SYSTEM_PROMPT
    assert ABSTAIN_TEXT in SYSTEM_PROMPT
    assert "exactly" in SYSTEM_PROMPT and "Cite only chunk ids" in SYSTEM_PROMPT


def test_context_chunks_appear_in_order_under_their_id_headers():
    e = comparison()
    text = format_context(e.context)
    positions = [text.index(f"[{cid}]\n") for cid in e.context_ids]
    assert positions == sorted(positions)
    assert text.count("\n\n[") == len(e.context) - 1


def test_user_message_ends_with_the_question():
    e = multi()
    assert user_message(e).endswith(f"Question: {e.task.question}")
    assert user_message(e).startswith("Context:\n\n[")


def test_prompt_never_contains_the_gold_answer_or_gold_labels():
    e = ex()
    text = SYSTEM_PROMPT + user_message(e)
    assert e.task.gold_answer not in text
    assert "gold" not in text.lower()


def test_prompt_is_identical_for_answerable_and_unanswerable_tasks_in_form():
    a, u = build_messages(ex()), build_messages(unanswerable())
    assert a[0] == u[0] and a[1]["content"].split("\n")[0] == u[1]["content"].split("\n")[0]


def test_prompt_is_deterministic_and_versioned():
    assert build_messages(ex()) == build_messages(ex())
    assert isinstance(PROMPT_VERSION, int) and PROMPT_VERSION >= 1


def test_chat_stub_renders_chatml_with_the_no_thinking_prefix():
    msgs = build_messages(ex())
    out = render_chat_stub(msgs)
    assert out.startswith("<|im_start|>system\n") and out.count("<|im_end|>") == 2
    assert out.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    thinking = render_chat_stub(msgs, enable_thinking=True)
    assert thinking.endswith("<|im_start|>assistant\n") and "<think>" not in thinking
    assert "<|im_start|>assistant" not in render_chat_stub(msgs, add_generation_prompt=False)


def test_example_context_is_a_plain_ordered_dict():
    e = Example(ex().task, {"b#001": "B", "a#001": "A"})
    assert list(e.context) == ["b#001", "a#001"]
