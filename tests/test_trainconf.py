from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from groundrl.trainconf import (
    ConfigError,
    apply_overrides,
    cli_overrides,
    load_config,
    set_path,
    validate_config,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = sorted((ROOT / "train" / "configs").glob("*.json"))
FIELDS = json.loads((Path(__file__).parent / "api_fields.json").read_text())


def raw(name="qwen3-0.6b-24gb"):
    return json.loads((ROOT / "train" / "configs" / f"{name}.json").read_text())


def test_there_are_configs_for_both_models_at_both_vram_profiles():
    names = {p.stem for p in CONFIGS}
    assert names == {"qwen3-0.6b-24gb", "qwen3-0.6b-12gb", "qwen3-1.7b-24gb", "qwen3-1.7b-12gb"}


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_shipped_configs_validate_and_describe_themselves(path):
    cfg = load_config(path)
    assert cfg["name"] == path.stem
    assert cfg["model"]["name_or_path"].startswith("Qwen/Qwen3-")
    assert cfg["profile"]["vram_gb"] == (12 if path.stem.endswith("12gb") else 24)
    assert cfg["grpo"]["reward"] == "v2"
    assert cfg["profile"]["estimates_only"] is True
    assert "none of them has been run" in cfg["description"]


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_sft_and_grpo_args_are_real_trl_and_transformers_fields(path):
    cfg = load_config(path)
    base = set(FIELDS["TrainingArguments"]) | set(FIELDS["TRL_BaseConfig"])
    assert set(cfg["sft"]["args"]) <= base | set(FIELDS["SFTConfig"])
    assert set(cfg["grpo"]["args"]) <= base | set(FIELDS["GRPOConfig"])


def test_lora_and_runner_kwargs_are_real_peft_fields():
    peft_fields = set(FIELDS["LoraConfig"]) | set(FIELDS["PeftConfig"])
    assert {"r", "lora_alpha", "lora_dropout", "target_modules", "bias", "task_type"} <= peft_fields


def test_twelve_gb_profiles_use_a_smaller_prompt_budget_and_group_and_the_4bit_one_is_1_7b():
    small, big = load_config(CONFIGS[0].parent / "qwen3-0.6b-12gb.json"), load_config(
        CONFIGS[0].parent / "qwen3-0.6b-24gb.json")
    assert small["data"]["max_prompt_tokens"] < big["data"]["max_prompt_tokens"]
    assert small["grpo"]["args"]["num_generations"] < big["grpo"]["args"]["num_generations"]
    q = load_config(CONFIGS[0].parent / "qwen3-1.7b-12gb.json")
    assert q["model"]["load_in_4bit"] is True and q["sft"]["args"]["optim"].startswith("paged")
    assert load_config(CONFIGS[0].parent / "qwen3-1.7b-24gb.json")["model"]["load_in_4bit"] is False


def test_defaults_are_filled_in():
    cfg = raw()
    for key in ("eval", "description"):
        cfg.pop(key, None)
    cfg["lora"].pop("dropout")
    out = validate_config(cfg)
    assert out["eval"]["split"] == "test" and out["lora"]["dropout"] == 0.05


def test_the_input_is_not_modified():
    cfg = raw()
    snapshot = copy.deepcopy(cfg)
    validate_config(cfg)
    assert cfg == snapshot


@pytest.mark.parametrize("mutate,message", [
    (lambda c: c.update(bogus=1), "unknown top-level"),
    (lambda c: c.pop("grpo"), "missing section"),
    (lambda c: c["model"].update(typo=1), "unknown key"),
    (lambda c: c["model"].update(load_in_4bit="yes"), "expected"),
    (lambda c: c["lora"].update(r=True), "expected"),
    (lambda c: c["lora"].update(r=0), "r and alpha"),
    (lambda c: c["lora"].update(dropout=1.0), "dropout"),
    (lambda c: c["model"].update(dtype="int4"), "dtype"),
    (lambda c: c["grpo"].update(reward="v9"), "reward"),
    (lambda c: c["data"].update(max_prompt_tokens=10), "too small"),
    (lambda c: c["eval"].update(temperature=-1.0), "eval"),
    (lambda c: c["sft"]["args"].update(max_length=100), "max_length"),
    (lambda c: c["grpo"]["args"].update(max_completion_length=64), "must equal"),
    (lambda c: c["grpo"]["args"].update(num_generations=1), "num_generations"),
    (lambda c: c["grpo"]["args"].update(num_generations=3), "divisible"),
    (lambda c: c["grpo"]["args"].update(beta=-0.1), "beta"),
    (lambda c: c["grpo"]["args"].update(temperature=0), "temperature"),
    (lambda c: c["grpo"]["args"].update(remove_unused_columns=True), "example_json"),
    (lambda c: c["grpo"]["args"].update(learning_rate=2.0), "learning_rate"),
    (lambda c: c["sft"]["args"].update(warmup_ratio=0.1), "warmup_steps"),
    (lambda c: c["sft"]["args"].update(warmup_steps=-1), "warmup_steps"),
    (lambda c: c["sft"]["args"].update(output_dir="x"), "output_dir"),
    (lambda c: c["sft"]["args"].update(per_device_train_batch_size=0), "per_device_train"),
    (lambda c: c["sft"]["args"].update(seed="a"), "seed"),
    (lambda c: c.update(name=""), "name"),
    (lambda c: c["profile"].update(vram_gb="big"), "expected"),
])
def test_invalid_configs_are_rejected_with_a_clear_message(mutate, message):
    cfg = raw()
    mutate(cfg)
    with pytest.raises(ConfigError, match=message):
        validate_config(cfg)


def test_missing_required_key_is_reported():
    cfg = raw()
    del cfg["lora"]["alpha"]
    with pytest.raises(ConfigError, match="missing required"):
        validate_config(cfg)


def test_invalid_json_file(tmp_path):
    bad = tmp_path / "c.json"
    bad.write_text("{not json")
    with pytest.raises(ConfigError, match="not valid JSON"):
        load_config(bad)


def test_set_path_and_errors():
    cfg = raw()
    set_path(cfg, "grpo.args.beta", 0.5)
    assert cfg["grpo"]["args"]["beta"] == 0.5
    with pytest.raises(ConfigError, match="no section"):
        set_path(cfg, "nope.x", 1)
    with pytest.raises(ConfigError, match="does not point"):
        set_path(cfg, "name.x", 1)


def test_apply_overrides_revalidates_and_does_not_mutate():
    cfg = load_config(ROOT / "train" / "configs" / "qwen3-0.6b-24gb.json")
    snapshot = copy.deepcopy(cfg)
    out = apply_overrides(cfg, {"grpo.args.beta": 0.1, "grpo.args.num_generations": 4})
    assert out["grpo"]["args"]["beta"] == 0.1 and cfg == snapshot
    with pytest.raises(ConfigError, match="divisible"):
        apply_overrides(cfg, {"grpo.args.num_generations": 5})


def test_cli_overrides_translate_flags():
    cfg = load_config(ROOT / "train" / "configs" / "qwen3-0.6b-24gb.json")
    ov = cli_overrides(cfg, seed=7, lr=1e-4, group_size=4, kl_beta=0.1, max_len=2000,
                       sets=["grpo.args.epsilon=0.3", "name=x y"], stage="grpo")
    assert ov["sft.args.seed"] == ov["grpo.args.seed"] == 7
    assert ov["grpo.args.learning_rate"] == 1e-4 and ov["grpo.args.num_generations"] == 4
    assert ov["grpo.args.beta"] == 0.1 and ov["data.max_prompt_tokens"] == 2000 - 320
    assert ov["sft.args.max_length"] == 2000
    assert ov["grpo.args.epsilon"] == 0.3 and ov["name"] == "x y"
    assert cli_overrides(cfg, lr=1e-3, stage="sft") == {"sft.args.learning_rate": 1e-3}
    assert cli_overrides(cfg) == {}


def test_cli_overrides_reject_bad_input():
    cfg = load_config(ROOT / "train" / "configs" / "qwen3-0.6b-24gb.json")
    with pytest.raises(ConfigError, match="leaves no room"):
        cli_overrides(cfg, max_len=300)
    with pytest.raises(ConfigError, match="key=value"):
        cli_overrides(cfg, sets=["oops"])
