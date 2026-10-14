from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from groundrl import trainctl
from groundrl.trainctl import MOCK_BANNER, build_parser, main

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = sorted((ROOT / "train" / "configs").glob("*.json"))
BIG = str(ROOT / "train" / "configs" / "qwen3-0.6b-24gb.json")


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
@pytest.mark.parametrize("stage", ["sft", "grpo", "eval"])
def test_dry_run_returns_zero_and_prints_the_plan(path, stage, capsys):
    rc = main(stage, ["--config", str(path), "--dry-run", "--repo-root", str(ROOT)])
    out = capsys.readouterr().out
    assert rc == 0 and out.startswith(f"DRY RUN: {stage} / {path.stem}") and "no warnings" in out


def test_dry_run_does_not_import_torch_in_a_fresh_interpreter():
    code = (
        "import sys; sys.path.insert(0, 'src');"
        "from groundrl.trainctl import main;"
        f"rc = main('grpo', ['--config', {BIG!r}, '--dry-run']);"
        "heavy = ('torch', 'transformers', 'trl', 'peft', 'datasets', 'vllm');"
        "bad = [m for m in heavy if m in sys.modules];"
        "print('HEAVY', bad); raise SystemExit(rc)"
    )
    done = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert "HEAVY []" in done.stdout


@pytest.mark.parametrize("stage", ["sft", "grpo", "eval"])
def test_entry_scripts_run_a_dry_run_for_the_two_main_configs(stage):
    for name in ("qwen3-0.6b-24gb", "qwen3-1.7b-12gb"):
        done = subprocess.run(
            [sys.executable, f"train/{stage}.py", "--config", f"train/configs/{name}.json",
             "--dry-run"], cwd=ROOT, capture_output=True, text=True)
        assert done.returncode == 0, done.stderr
        assert "DRY RUN" in done.stdout and "memory_estimate_gb" in done.stdout


def test_flags_override_the_config_before_planning(capsys):
    rc = main("grpo", ["--config", BIG, "--dry-run", "--repo-root", str(ROOT), "--group-size",
                       "4", "--kl-beta", "0.1", "--lr", "0.0001", "--seed", "5", "--max-len",
                       "2000"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "group_size: 4" in out and "kl_beta: 0.1" in out and "learning_rate: 0.0001" in out
    assert "max_prompt_tokens: 1680" in out


def test_set_flag_and_limit(capsys):
    rc = main("sft", ["--config", BIG, "--dry-run", "--repo-root", str(ROOT), "--limit", "40",
                      "--set", "sft.args.num_train_epochs=3"])
    out = capsys.readouterr().out
    assert rc == 0 and "examples: 40" in out and "epochs: 3" in out


def test_bad_overrides_are_reported_as_config_errors(capsys):
    rc = main("grpo", ["--config", BIG, "--dry-run", "--group-size", "5"])
    assert rc == 2 and "divisible" in capsys.readouterr().err


def test_missing_config_file_is_a_config_error(capsys):
    assert main("sft", ["--config", "/nonexistent/config.json", "--dry-run"]) == 2
    assert "config error" in capsys.readouterr().err


def test_unknown_stage():
    with pytest.raises(ValueError, match="unknown stage"):
        main("distill", [])


def test_parser_has_stage_specific_flags():
    assert build_parser("grpo").parse_args(["--config", "c", "--group-size", "4",
                                            "--init-from", "m"]).init_from == "m"
    assert build_parser("sft").parse_args(["--config", "c", "--no-merge"]).no_merge
    ns = build_parser("eval").parse_args(["--config", "c", "--model", "m", "--adapter", "a",
                                          "--split", "val", "--label", "x"])
    assert (ns.model, ns.adapter, ns.split, ns.label) == ("m", "a", "val", "x")


def test_mock_runs_are_labelled_and_never_write_to_results(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    common = ["--config", BIG, "--mock", "--repo-root", str(ROOT), "--limit", "16"]
    assert main("sft", common) == 0
    out = capsys.readouterr().out
    assert MOCK_BANNER in out and '"mock": true' in out
    assert main("grpo", common + ["--init-from", "outputs/mock/qwen3-0.6b-24gb/sft/merged"]) == 0
    assert main("eval", ["--config", BIG, "--mock", "--repo-root", str(ROOT), "--limit", "10",
                         "--adapter", "x"]) == 0
    mock_dir = tmp_path / "outputs" / "mock"
    written = list(mock_dir.rglob("*.json"))
    assert any(p.name.startswith("eval_") for p in written)
    for p in written:
        data = json.loads(p.read_text())
        if "mock" in data:
            assert data["mock"] is True, p
    assert not (tmp_path / "results").exists()


def test_mock_stack_policy_returns_empty_for_unknown_prompts():
    from groundrl.trainconf import load_config

    stack = trainctl._mock_stack(load_config(BIG), ROOT)
    tok = stack.AutoTokenizer.from_pretrained("x")
    model = stack.AutoModelForCausalLM.from_pretrained("x")
    enc = tok(["definitely not a dataset prompt"], return_tensors="pt", padding=True)
    out = model.generate(**enc)
    assert tok.batch_decode(out[:, enc["input_ids"].shape[1]:]) == [""]
