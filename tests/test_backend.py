"""MOCK runs: these exercise trainer plumbing with a fake stack; nothing is trained."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from groundrl import trl_backend
from groundrl.audit import POLICY_BY_NAME, run_policy
from groundrl.dataset import load_dataset
from groundrl.example import make_examples
from groundrl.mockstack import make_mock_stack
from groundrl.prompts import build_messages, render_chat_stub
from groundrl.trainconf import apply_overrides, load_config

ROOT = Path(__file__).resolve().parents[1]
FIELDS = json.loads((Path(__file__).parent / "api_fields.json").read_text())
STRICT = {
    "SFTConfig": (set(FIELDS["SFTConfig"]) | set(FIELDS["TRL_BaseConfig"])
                  | set(FIELDS["TrainingArguments"])),
    "GRPOConfig": (set(FIELDS["GRPOConfig"]) | set(FIELDS["TRL_BaseConfig"])
                   | set(FIELDS["TrainingArguments"])),
    "LoraConfig": set(FIELDS["LoraConfig"]) | set(FIELDS["PeftConfig"]),
}


@pytest.fixture(scope="module")
def prompt_lookup():
    ds = load_dataset(ROOT / "data")
    out = {}
    for split in ("train", "val", "test"):
        for ex in make_examples(ds.tasks[split], ds.index):
            out[render_chat_stub(build_messages(ex), True, False)] = ex
    return out


def stack_for(lookup, record):
    spec = POLICY_BY_NAME["extractive"]

    def policy(prompt):
        ex = lookup.get(prompt)
        return run_policy(spec, ex) if ex is not None else ""

    return make_mock_stack(policy, strict_fields=STRICT, record=record)


def small_cfg(name="qwen3-0.6b-24gb", **extra):
    cfg = load_config(ROOT / "train" / "configs" / f"{name}.json")
    return apply_overrides(cfg, {"data.limit_train": 24, "data.limit_eval": 8,
                                 "data.data_dir": str(ROOT / "data"), **extra})


def events(record, name):
    return [p for e, p in record if e == name]


def test_sft_mock_run_wires_config_dataset_and_outputs(tmp_path, prompt_lookup):
    record = []
    cfg = small_cfg()
    summary = trl_backend.run_sft(cfg, tmp_path, stack_for(prompt_lookup, record), mock=True)
    assert summary["mock"] is True and summary["stage"] == "sft"
    sft_cfg = events(record, "SFTConfig")[0]
    assert sft_cfg["output_dir"] == str(tmp_path / "trainer")
    assert sft_cfg["learning_rate"] == cfg["sft"]["args"]["learning_rate"]
    assert sft_cfg["completion_only_loss"] is True and sft_cfg["max_length"] == 2560
    lora = events(record, "LoraConfig")[0]
    assert lora == {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05,
                    "target_modules": "all-linear", "bias": "none", "task_type": "CAUSAL_LM"}
    assert events(record, "SFTTrainer")[0]["peft_config"] == "LoraConfig"
    assert (tmp_path / "adapter" / "adapter_config.json").exists()
    assert (tmp_path / "merged" / "config.json").exists()
    written = json.loads((tmp_path / "summary_sft.json").read_text())
    assert written["mock"] is True and written["train_examples"] > 0
    assert written["merged_dir"].endswith("merged")


def test_sft_rows_are_pre_rendered_prompts_with_the_gold_answer(tmp_path, prompt_lookup):
    captured = {}
    stack = stack_for(prompt_lookup, [])
    original = stack.SFTTrainer

    class Spy(original):
        def __init__(self, **kw):
            captured.update(kw)
            super().__init__(**kw)

    stack.SFTTrainer = Spy
    trl_backend.run_sft(small_cfg(), tmp_path, stack, mock=True, merge=False)
    row = captured["train_dataset"][0]
    assert set(row) == {"prompt", "completion"}
    assert row["prompt"].endswith("<think>\n\n</think>\n\n") and "[eip-" in row["prompt"]
    assert prompt_lookup[row["prompt"]].task.gold_answer == row["completion"]


def test_sft_without_merge_skips_the_merged_model(tmp_path, prompt_lookup):
    summary = trl_backend.run_sft(small_cfg(), tmp_path, stack_for(prompt_lookup, []), mock=True,
                                  merge=False)
    assert summary["merged_dir"] is None and not (tmp_path / "merged").exists()


def test_merge_reloads_the_base_unquantised_even_for_4bit_runs(tmp_path, prompt_lookup):
    record = []
    cfg = small_cfg("qwen3-1.7b-12gb")
    trl_backend.run_sft(cfg, tmp_path, stack_for(prompt_lookup, record), mock=True)
    loads = events(record, "model")
    assert "quantization_config" in loads[0] and loads[0]["device_map"] == {"": 0}
    assert "quantization_config" not in loads[-1]  # the merge reload


def test_grpo_mock_run_uses_the_reward_wrapper_and_example_column(tmp_path, prompt_lookup):
    record = []
    cfg = small_cfg()
    summary = trl_backend.run_grpo(cfg, tmp_path, stack_for(prompt_lookup, record), mock=True)
    assert summary["mock"] is True and summary["reward"] == "v2" and summary["group_size"] == 8
    grpo_cfg = events(record, "GRPOConfig")[0]
    assert grpo_cfg["remove_unused_columns"] is False and grpo_cfg["num_generations"] == 8
    assert grpo_cfg["steps_per_generation"] == 4 and grpo_cfg["beta"] == 0.02
    trainer = events(record, "GRPOTrainer")[0]
    assert trainer["reward_funcs"] == "function" and trainer["peft_config"] == "LoraConfig"
    log = json.loads((tmp_path / "summary_grpo.json").read_text())["log_history"]
    assert log[0]["mock"] == 1 and 0.0 <= log[0]["mock_scripted_reward_mean"] <= 1.0


def test_grpo_starts_from_the_merged_sft_model_when_asked(tmp_path, prompt_lookup):
    record = []
    cfg = small_cfg(**{"model.init_from": "outputs/sft/merged"})
    trl_backend.run_grpo(cfg, tmp_path, stack_for(prompt_lookup, record), mock=True)
    assert events(record, "model")[0]["path"] == "outputs/sft/merged"
    assert events(record, "tokenizer")[0]["path"] == "outputs/sft/merged"


def test_prompts_over_the_token_budget_are_dropped_from_training(tmp_path, prompt_lookup):
    cfg = small_cfg(**{"data.max_prompt_tokens": 1100, "data.limit_train": 30})
    summary = trl_backend.run_grpo(cfg, tmp_path, stack_for(prompt_lookup, []), mock=True)
    assert summary["dropped_by_prompt_budget"] > 0
    assert summary["train_examples"] + summary["dropped_by_prompt_budget"] == 30


def test_a_budget_that_drops_every_prompt_is_an_error(tmp_path, prompt_lookup):
    cfg = small_cfg(**{"data.max_prompt_tokens": 100})
    with pytest.raises(ValueError, match="no training prompt fits"):
        trl_backend.run_grpo(cfg, tmp_path / "g", stack_for(prompt_lookup, []), mock=True)
    with pytest.raises(ValueError, match="no training prompt fits"):
        trl_backend.run_sft(cfg, tmp_path / "s", stack_for(prompt_lookup, []), mock=True)


def test_eval_mock_run_writes_a_labelled_result_with_metrics(tmp_path, prompt_lookup):
    record = []
    out = tmp_path / "eval_mock.json"
    result = trl_backend.run_eval(small_cfg(), None, "outputs/grpo/adapter", out,
                                  stack=stack_for(prompt_lookup, record), mock=True, limit=20,
                                  label="mock-run")
    on_disk = json.loads(out.read_text())
    assert on_disk["kind"] == "eval" and on_disk["mock"] is True and on_disk["label"] == "mock-run"
    assert on_disk["n"] == 20 and "grounded_correct_rate" in on_disk["metrics"]
    assert on_disk["metrics"]["length_tokens"]["max"] >= 1
    assert events(record, "peft_model")[0]["path"] == "outputs/grpo/adapter"
    gen = events(record, "generate")
    assert gen and gen[0]["do_sample"] is False and gen[0]["max_new_tokens"] == 320
    assert result["samples"][0]["id"].startswith("test-")


def test_eval_sampling_is_enabled_only_for_positive_temperature(tmp_path, prompt_lookup):
    record = []
    cfg = small_cfg(**{"eval.temperature": 0.7})
    trl_backend.run_eval(cfg, None, None, tmp_path / "e.json", stack=stack_for(
        prompt_lookup, record), mock=True, limit=4)
    gen = events(record, "generate")[0]
    assert gen["do_sample"] is True and gen["temperature"] == 0.7


def test_generation_batches_prompts_and_returns_only_new_text(prompt_lookup):
    stack = stack_for(prompt_lookup, [])
    tok = stack.AutoTokenizer.from_pretrained("x")
    model = stack.AutoModelForCausalLM.from_pretrained("x")
    exs = list(prompt_lookup.values())[:5]
    prompts = list(prompt_lookup)[:5]
    out = trl_backend.generate_completions(stack, model, tok, prompts, batch_size=2,
                                           max_new_tokens=10, temperature=0.0)
    assert len(out) == 5
    spec = POLICY_BY_NAME["extractive"]
    assert out == [run_policy(spec, e) for e in exs]


def test_load_stack_without_the_training_extra_explains_how_to_install():
    try:
        import torch  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match=r"pip install -e '\.\[train\]'"):
            trl_backend.load_stack()
    else:  # pragma: no cover - only when the training extra happens to be installed
        pytest.skip("training extra is installed")


def test_environment_info_and_git_helpers(prompt_lookup):
    stack = stack_for(prompt_lookup, [])
    info = trl_backend.environment_info(stack)
    assert info["torch"] == "mock" and info["gpu"] is None
    broken = stack_for(prompt_lookup, [])
    broken.torch = None
    assert trl_backend.environment_info(broken)["gpu"] == "unknown"
    commit = trl_backend.git_commit(ROOT)
    assert commit is None or len(commit) == 40
    assert trl_backend.git_commit(Path("/nonexistent-dir-for-test")) is None


def test_numeric_log_filters_non_numeric_entries():
    rows = [{"step": 1, "loss": 0.5, "note": "x", "flag": True}]
    assert trl_backend._numeric_log(rows) == [{"step": 1, "loss": 0.5, "flag": True}]


def test_strict_mock_rejects_unknown_config_keys(prompt_lookup):
    stack = stack_for(prompt_lookup, [])
    with pytest.raises(TypeError, match="unexpected keyword"):
        stack.GRPOConfig(output_dir="x", not_a_real_trl_option=1)
    with pytest.raises(TypeError, match="unexpected keyword"):
        stack.LoraConfig(r=8, lora_alphaa=16)


def test_all_shipped_configs_run_through_the_mock_stack(tmp_path, prompt_lookup):
    for name in ("qwen3-0.6b-24gb", "qwen3-0.6b-12gb", "qwen3-1.7b-24gb", "qwen3-1.7b-12gb"):
        cfg = small_cfg(name)
        trl_backend.run_sft(cfg, tmp_path / name / "sft", stack_for(prompt_lookup, []), True)
        trl_backend.run_grpo(cfg, tmp_path / name / "grpo", stack_for(prompt_lookup, []), True)
