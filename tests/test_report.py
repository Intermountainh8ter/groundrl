from __future__ import annotations

import json
import runpy
import sys

import pytest

from groundrl.audit import audit
from groundrl.evaluate import evaluate_completions
from groundrl.report import NOT_RUN, _fmt, eval_table, load_results, render_report
from tests.test_rewards import GOLD, ex, unanswerable


def eval_blob(label="run-a", mock=False):
    exs = [ex(), unanswerable()]
    body = evaluate_completions(exs, [GOLD, "I cannot answer this from the provided context."],
                                n_boot=20)
    body.update({"kind": "eval", "mock": mock, "label": label, "split": "test",
                 "model_path": "Qwen/Qwen3-0.6B", "adapter_dir": "out/adapter",
                 "git_commit": "abc123"})
    return body


def write(path, payload):
    path.write_text(json.dumps(payload))


def test_without_any_real_eval_the_report_says_not_yet_run(tmp_path):
    text = render_report(tmp_path)
    assert NOT_RUN in text and "No audit file found" in text and "No baseline file" in text


def test_committed_results_directory_does_not_claim_training_results(root):
    text = render_report(root / "results")
    assert NOT_RUN in text
    assert "Reward-hacking audit" in text and "Scripted baselines" in text


def test_audit_and_baseline_sections_render_from_their_files(tmp_path):
    result = audit([ex(), unanswerable()], n_boot=10)
    result.update(split="test", command="python scripts/reward_audit.py --split test")
    write(tmp_path / "audit_test.json", result)
    baseline = {"kind": "baseline_eval", "mock": False, "split": "test",
                "policies": {"oracle": eval_blob("x")}}
    write(tmp_path / "baseline_test.json", baseline)
    text = render_report(tmp_path)
    assert "Reward-hacking audit on the TEST split" in text
    assert "scripts/reward_audit.py --split test" in text
    assert "`oracle`" in text and "not trained models" in text


def test_a_real_eval_replaces_not_yet_run_with_a_table(tmp_path):
    write(tmp_path / "eval_a.json", eval_blob("run-a"))
    text = render_report(tmp_path)
    assert NOT_RUN not in text
    assert "`run-a`" in text and "Qwen/Qwen3-0.6B" in text and "abc123" in text


def test_mock_files_are_listed_as_ignored_and_never_counted(tmp_path):
    write(tmp_path / "eval_mock.json", eval_blob("mock-run", mock=True))
    write(tmp_path / "train_mock.json", {"kind": "train", "mock": True, "stage": "grpo"})
    text = render_report(tmp_path)
    assert NOT_RUN in text
    assert "Ignored MOCK files" in text and "eval_mock.json" in text
    assert "`mock-run`" not in text


def test_train_runs_are_listed(tmp_path):
    write(tmp_path / "train_grpo.json", {"kind": "train", "mock": False, "stage": "grpo",
                                         "config": "c", "seed": 3, "log_history": [{}, {}]})
    text = render_report(tmp_path)
    assert "stage grpo" in text and "2 log rows" in text and NOT_RUN in text


def test_unreadable_and_non_object_json_are_tolerated(tmp_path):
    (tmp_path / "bad.json").write_text("{nope")
    (tmp_path / "list.json").write_text("[1, 2]")
    kinds = [d["kind"] for _, d in load_results(tmp_path)]
    assert kinds == ["unreadable", "unreadable"]
    assert NOT_RUN in render_report(tmp_path)


@pytest.mark.parametrize("metric,pct,expected", [
    ({"mean": 0.5, "lo": 0.4, "hi": 0.6}, False, "0.500 (0.400-0.600)"),
    ({"value": 0.5, "lo": 0.4, "hi": 0.6}, True, "50.0% (40.0-60.0)"),
    ({"value": None}, True, "n/a"),
    ({"value": 0.25, "lo": None, "hi": None}, True, "25.0%"),
    (None, False, "n/a"),
])
def test_metric_formatting(metric, pct, expected):
    assert _fmt(metric, pct) == expected


def test_eval_table_has_one_row_per_run():
    table = eval_table([("a", eval_blob()), ("b", eval_blob())])
    assert table.count("\n") == 3 and "| `a` |" in table and "grounded-correct" in table


def test_make_report_script_writes_the_file(tmp_path, root, monkeypatch, capsys):
    write(tmp_path / "eval_a.json", eval_blob("run-a"))
    monkeypatch.setattr(sys, "argv", ["make_report.py", "--results", str(tmp_path)])
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(root / "scripts" / "make_report.py"), run_name="__main__")
    assert exc.value.code == 0
    assert "`run-a`" in (tmp_path / "REPORT.md").read_text()
    assert "# groundrl report" in capsys.readouterr().out


def test_runbook_quotes_the_committed_baseline_numbers(root):
    import re

    baseline = json.loads((root / "results" / "baseline_test.json").read_text())
    ext = baseline["policies"]["extractive"]["metrics"]
    runbook = (root / "docs" / "RUNBOOK.md").read_text()
    v2 = f"{ext['mean_reward']['v2']['mean']:.3f}"
    gc = f"{ext['grounded_correct_rate']['mean'] * 100:.1f}%"
    pattern = rf"scores {re.escape(v2)} reward v2 and {re.escape(gc)} grounded-correct"
    assert re.search(pattern, runbook)
