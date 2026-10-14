"""Render ``results/*.json`` into a Markdown report. Renders only what is on disk.

File kinds (the ``kind`` field of each JSON file):

* ``reward_audit``   scripted-policy audit of the reward versions (real, needs no model)
* ``baseline_eval``  scripted baseline policies evaluated with the model pipeline (not trained
                     models; reference points)
* ``eval``           a trained (or base) model evaluated on a split, produced by ``train/eval.py``
* ``train``          a training run summary produced by ``train/sft.py`` or ``train/grpo.py``

Any file with ``"mock": true`` is MOCK output from the plumbing tests. It is listed as ignored and
never contributes a number. When no real ``eval`` file exists the report says
"Training results: NOT YET RUN"; nothing is invented or filled in.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .audit import render_markdown

NOT_RUN = "Training results: NOT YET RUN"


def load_results(results_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    out = []
    for path in sorted(Path(results_dir).glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {"kind": "unreadable"}
        out.append((path, data if isinstance(data, dict) else {"kind": "unreadable"}))
    return out


def _fmt(metric: Any, pct: bool = False) -> str:
    if not isinstance(metric, dict) or metric.get("value", metric.get("mean")) is None:
        return "n/a"
    v = metric.get("value", metric.get("mean"))
    lo, hi = metric.get("lo"), metric.get("hi")
    scale = 100 if pct else 1
    unit = "%" if pct else ""
    digits = 1 if pct else 3
    txt = f"{v * scale:.{digits}f}{unit}"
    if lo is not None and hi is not None:
        txt += f" ({lo * scale:.{digits}f}-{hi * scale:.{digits}f})"
    return txt


EVAL_COLUMNS = (
    ("grounded-correct", "grounded_correct_rate", True),
    ("correct", "correct_rate", True),
    ("verified", "verified_rate", True),
    ("abstain precision", "abstention_precision", True),
    ("abstain recall", "abstention_recall", True),
    ("hallucinated cite", "hallucinated_citation_rate", True),
)


def eval_table(rows: list[tuple[str, dict[str, Any]]]) -> str:
    head = ["run", "n"] + [c[0] for c in EVAL_COLUMNS] + ["v0", "v1", "v2", "words (mean)"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for label, data in rows:
        m = data["metrics"]
        cells = [f"`{label}`", str(data["n"])]
        cells += [_fmt(m.get(key), pct) for _, key, pct in EVAL_COLUMNS]
        cells += [_fmt(m["mean_reward"].get(v)) for v in ("v0", "v1", "v2")]
        cells.append(f"{m['length_words']['mean']:.1f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_report(results_dir: Path) -> str:
    files = load_results(results_dir)
    mock = [p.name for p, d in files if d.get("mock")]
    real = [(p, d) for p, d in files if not d.get("mock")]
    audits = [(p, d) for p, d in real if d.get("kind") == "reward_audit"]
    baselines = [(p, d) for p, d in real if d.get("kind") == "baseline_eval"]
    evals = [(p, d) for p, d in real if d.get("kind") == "eval"]
    trains = [(p, d) for p, d in real if d.get("kind") == "train"]
    out: list[str] = ["# groundrl report", "",
                      "Rendered by `scripts/make_report.py` from `results/*.json` only.", ""]
    out.append("## Reward-hacking audit (real, no model)")
    out.append("")
    if audits:
        for path, data in audits:
            out += [f"Source: `{path.name}`; regenerate with `{data.get('command', '')}`.", "",
                    render_markdown(data, data.get("split", "test")), ""]
    else:
        out += ["No audit file found. Run `python scripts/reward_audit.py --split test`.", ""]
    out.append("## Scripted baselines (not trained models)")
    out.append("")
    if baselines:
        for path, data in baselines:
            rows = [(name, ev) for name, ev in data["policies"].items()]
            out += [f"Source: `{path.name}` ({data['split']} split). Scripted policies, no "
                    "learning.", "", eval_table(rows), ""]
    else:
        out += ["No baseline file found. Run `groundrl baseline-eval --split test`.", ""]
    out.append("## Training results")
    out.append("")
    if evals:
        rows = [(d.get("label", p.stem), d) for p, d in evals]
        out += [eval_table(rows), ""]
        for path, d in evals:
            out.append(f"* `{path.name}`: split {d.get('split')}, model `{d.get('model_path')}`, "
                       f"adapter `{d.get('adapter_dir')}`, commit {d.get('git_commit')}")
        out.append("")
    else:
        out += [f"**{NOT_RUN}.** No `kind: eval` result file from a real run exists in "
                f"`{Path(results_dir)}`.", ""]
    if trains:
        out.append("Training runs on record:")
        out.append("")
        for path, d in trains:
            out.append(f"* `{path.name}`: stage {d.get('stage')}, config {d.get('config')}, seed "
                       f"{d.get('seed')}, {len(d.get('log_history', []))} log rows")
        out.append("")
    if mock:
        out += ["## Ignored MOCK files", "",
                "These come from plumbing tests with a fake model and are not results: "
                + ", ".join(f"`{n}`" for n in mock), ""]
    return "\n".join(out).rstrip() + "\n"
