from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from groundrl.example import Example
from groundrl.prompts import build_messages
from groundrl.trainconf import apply_overrides, load_config
from groundrl.trainplan import (
    StubTokenizer,
    build_plan,
    estimate_memory_gb,
    format_plan,
    grpo_record,
    model_params_b,
    prompt_text,
    sft_record,
)
from tests.test_rewards import ex

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = sorted((ROOT / "train" / "configs").glob("*.json"))


def test_stub_tokenizer_counts_words_digits_hex_and_symbols():
    t = StubTokenizer()
    assert t.count("") == 0
    assert t.count("hello") == 1 and t.count("hello world") == 2
    assert t.count("abcdefghij") == 2  # five letters per token
    assert t.count("4096") == 2 and t.count("123") == 1
    assert t.count("0x03") == 2 and t.count("0xdeadbeef") == 5
    assert t.count("a.b") == 3
    assert t.count("FIELD_ELEMENTS_PER_BLOB") == 5


def test_stub_render_matches_the_chat_stub_and_respects_thinking():
    t = StubTokenizer()
    msgs = build_messages(ex())
    assert t.render(msgs).endswith("<think>\n\n</think>\n\n")
    assert not t.render(msgs, enable_thinking=True).endswith("</think>\n\n")
    assert prompt_text(ex(), t.render) == t.render(msgs)


def test_sft_and_grpo_records():
    e = ex()
    sft = sft_record(e, "PROMPT")
    assert sft == {"prompt": "PROMPT", "completion": e.task.gold_answer}
    row = grpo_record(e, "PROMPT")
    assert set(row) == {"prompt", "example_json"}
    assert Example.from_json(row["example_json"]).task == e.task


def test_memory_estimate_is_monotone_and_4bit_is_smaller():
    base = dict(params_b=1.7, lora_r=16, prompt_tokens=1536, completion_tokens=320, stage="sft")
    small = estimate_memory_gb(load_in_4bit=False, micro_batch=1, **base)
    big = estimate_memory_gb(load_in_4bit=False, micro_batch=4, **base)
    q4 = estimate_memory_gb(load_in_4bit=True, micro_batch=1, **base)
    assert q4 < small < big
    grpo_small = estimate_memory_gb(1.7, False, 16, 1536, 320, 2, "grpo", gen_batch=8)
    grpo_big = estimate_memory_gb(1.7, False, 16, 1536, 320, 2, "grpo", gen_batch=64)
    assert grpo_small < grpo_big
    assert estimate_memory_gb(1.7, False, 16, 1536, 320, 8, "eval") > 3.0
    assert estimate_memory_gb(30.0, False, 16, 100, 100, 1, "sft") > 0  # nearest architecture


def test_model_params_from_name():
    assert model_params_b("Qwen/Qwen3-0.6B") == 0.6 and model_params_b("Qwen/Qwen3-1.7B") == 1.7
    assert model_params_b("something-else") == 1.0


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
@pytest.mark.parametrize("stage", ["sft", "grpo", "eval"])
def test_dry_run_plans_every_config_and_stage(path, stage):
    cfg = load_config(path)
    plan = build_plan(cfg, stage, repo_root=ROOT)
    assert plan["stage"] == stage and plan["config"] == cfg["name"]
    assert plan["examples"] > 200 and 0 < plan["kept_after_prompt_budget"] <= plan["examples"]
    assert plan["memory_estimate_gb"] < 0.85 * cfg["profile"]["vram_gb"]
    assert plan["warnings"] == []
    assert "NOT measured" in plan["memory_estimate_note"]
    assert "approximate" in plan["tokenizer"]
    text = format_plan(plan)
    assert text.startswith(f"DRY RUN: {stage} / {cfg['name']}") and "no torch" in text


def test_dry_run_imports_no_heavy_module():
    """build_plan must not pull in torch/TRL: the dry run has to work with no GPU stack.

    This runs in a child interpreter on purpose. sys.modules is global to a
    pytest session, so once any other test imports torch the assertion below
    can never fail again -- it would silently stop testing anything.
    """
    source = textwrap.dedent(
        """
        import sys
        sys.path.insert(0, {src!r})
        from groundrl.trainconf import load_config
        from groundrl.trainplan import build_plan
        cfg = load_config({cfg!r})
        build_plan(cfg, "grpo", repo_root={root!r})
        heavy = [m for m in ("torch", "transformers", "trl", "peft", "datasets", "vllm")
                 if m in sys.modules]
        print(",".join(heavy))
        """
    ).format(src=str(ROOT / "src"), cfg=str(CONFIGS[0]), root=str(ROOT))
    r = subprocess.run([sys.executable, "-c", source], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "", f"dry run imported: {r.stdout.strip()}"


def test_sft_plan_counts_steps():
    cfg = load_config(ROOT / "train" / "configs" / "qwen3-0.6b-24gb.json")
    plan = build_plan(cfg, "sft", repo_root=ROOT)
    a = cfg["sft"]["args"]
    eff = a["per_device_train_batch_size"] * a["gradient_accumulation_steps"]
    assert plan["effective_batch"] == eff
    assert plan["planned_steps"] == -(-plan["steps_per_epoch"] * a["num_train_epochs"] // 1)
    assert plan["split"] == "train"


def test_grpo_plan_counts_prompts_and_groups():
    cfg = load_config(ROOT / "train" / "configs" / "qwen3-0.6b-24gb.json")
    plan = build_plan(cfg, "grpo", repo_root=ROOT)
    assert plan["group_size"] == 8 and plan["completions_per_optimizer_step"] == 64
    assert plan["prompts_per_optimizer_step"] == 8 and plan["generation_batch"] == 16
    assert plan["planned_steps"] == 300 and plan["reward"] == "v2" and plan["kl_beta"] == 0.02
    expected = 300 * 8 / plan["kept_after_prompt_budget"]
    assert plan["epochs_covered"] == pytest.approx(expected, abs=0.01)


def test_eval_plan_uses_the_test_split_and_keeps_every_prompt():
    cfg = load_config(ROOT / "train" / "configs" / "qwen3-0.6b-12gb.json")
    plan = build_plan(cfg, "eval", repo_root=ROOT)
    assert plan["split"] == "test" and plan["batch_size"] == 4
    assert plan["notes"] == []  # evaluation never drops prompts


def test_plan_warns_when_many_prompts_exceed_the_budget_and_when_memory_does_not_fit():
    cfg = load_config(ROOT / "train" / "configs" / "qwen3-0.6b-24gb.json")
    tight = apply_overrides(cfg, {"data.max_prompt_tokens": 700})
    plan = build_plan(tight, "grpo", repo_root=ROOT)
    assert any("exceed max_prompt_tokens" in w for w in plan["warnings"])
    heavy = apply_overrides(cfg, {"profile.vram_gb": 6})
    assert any("above 85%" in w for w in build_plan(heavy, "grpo", repo_root=ROOT)["warnings"])


def test_plan_notes_small_prompt_drops_without_warning():
    cfg = load_config(ROOT / "train" / "configs" / "qwen3-0.6b-12gb.json")
    plan = build_plan(cfg, "sft", repo_root=ROOT)
    assert plan["dropped"] > 0 and plan["warnings"] == []
    assert any("dropped from training" in n for n in plan["notes"])


def test_plan_warns_when_gold_answers_may_not_fit_the_completion_budget():
    cfg = load_config(ROOT / "train" / "configs" / "qwen3-0.6b-24gb.json")
    short = apply_overrides(cfg, {"data.max_completion_tokens": 64,
                                  "grpo.args.max_completion_length": 64})
    assert any("max_completion_tokens" in w for w in build_plan(short, "grpo",
                                                                 repo_root=ROOT)["warnings"])


def test_plan_limit_train_is_honoured():
    cfg = load_config(ROOT / "train" / "configs" / "qwen3-0.6b-24gb.json")
    limited = apply_overrides(cfg, {"data.limit_train": 40})
    assert build_plan(limited, "sft", repo_root=ROOT)["examples"] == 40


def test_unknown_stage_rejected():
    cfg = load_config(ROOT / "train" / "configs" / "qwen3-0.6b-24gb.json")
    with pytest.raises(ValueError, match="unknown stage"):
        build_plan(cfg, "pretrain", repo_root=ROOT)


def test_steps_per_generation_above_accumulation_is_flagged():
    cfg = load_config(ROOT / "train" / "configs" / "qwen3-0.6b-24gb.json")
    off = apply_overrides(cfg, {"grpo.args.steps_per_generation": 32})
    assert any("off-policy" in w for w in build_plan(off, "grpo", repo_root=ROOT)["warnings"])
