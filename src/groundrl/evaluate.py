"""Evaluation metrics for a set of completions, with bootstrap confidence intervals.

Pure functions: given examples and the completions a model (or a scripted policy) produced, compute

* ``grounded_correct_rate``  answerable tasks whose answer is correct (every gold token, no extra
  claim), accepted by the verifier and cites a gold support chunk
* ``correct_rate``           answerable tasks whose answer is correct, grounded or not
* ``verified_rate``          answerable tasks whose answer the verifier accepts
* ``abstention_precision`` / ``abstention_recall``  with "should abstain" = unanswerable task
* ``hallucinated_citation_rate``  completions citing a chunk that is not in the context
* mean reward under v0, v1 and v2
* answer length and format statistics

Every rate carries a 95% percentile-bootstrap interval over tasks. Nothing here needs a model.
"""
from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from statistics import median

import numpy as np

from .example import Example, make_examples
from .protocol import is_abstention
from .rewards import completion_text, correctness, reward_v0, reward_v1, reward_v2
from .stats import mean_ci, ratio_ci
from .tokens import word_count
from .verifier import verify_answer

REWARD_FNS = {"v0": reward_v0, "v1": reward_v1, "v2": reward_v2}


def _grounded_correct(ex: Example, completion: str) -> tuple[bool, bool, bool]:
    """``(correct, verified, grounded_correct)`` for one answerable task."""
    res = verify_answer(completion, ex.context, ex.context_ids)
    _, recall, _, extras = correctness(completion, ex, 0.25)
    correct = recall == 1.0 and extras == 0
    groups = [set(g) for g in ex.task.gold_groups if g]
    cited = set(res.cited_chunks)
    cites_gold = bool(groups) and all(cited & g for g in groups)
    return correct, res.ok, correct and res.ok and cites_gold


def evaluate_completions(
    examples: Sequence[Example], completions: Sequence[str], n_boot: int = 2000, seed: int = 0,
    token_counts: Sequence[int] | None = None,
) -> dict:
    """Metrics for ``completions[i]`` answering ``examples[i]``."""
    if len(examples) != len(completions):
        raise ValueError("one completion per example is required")
    texts = [completion_text(c) for c in completions]
    answerable = [e.task.answerable for e in examples]
    rows: list[dict] = []
    for ex, text in zip(examples, texts, strict=True):
        res = verify_answer(text, ex.context, ex.context_ids)
        correct, verified, grounded = _grounded_correct(ex, text) if ex.task.answerable \
            else (False, False, False)
        n_cites = sum(len(s.citations) for s in res.sentences)
        row = {
            "answerable": ex.task.answerable, "qtype": ex.task.qtype,
            "abstained": is_abstention(text), "correct": correct, "verified": verified,
            "grounded_correct": grounded, "invalid_cites": len(res.invalid_citations),
            "n_cites": n_cites, "n_invalid_cites": sum(len(s.invalid) for s in res.sentences),
            "near_miss": res.near_miss > 0, "suspicious": bool(res.suspicious),
            "empty": not text.strip(), "words": word_count(text),
        }
        for name, fn in REWARD_FNS.items():
            row[name] = fn(text, ex)[0]
        rows.append(row)

    def col(key: str, subset: Callable[[dict], bool] = lambda r: True) -> list[float]:
        return [float(r[key]) for r in rows if subset(r)]

    ans = lambda r: r["answerable"]  # noqa: E731
    metrics: dict[str, object] = {
        "grounded_correct_rate": mean_ci(col("grounded_correct", ans), n_boot, seed),
        "correct_rate": mean_ci(col("correct", ans), n_boot, seed),
        "verified_rate": mean_ci(col("verified", ans), n_boot, seed),
        "hallucinated_citation_rate": mean_ci(
            [float(r["invalid_cites"] > 0) for r in rows], n_boot, seed),
        "format_drift_rate": mean_ci(col("near_miss"), n_boot, seed),
        "suspicious_character_rate": mean_ci(col("suspicious"), n_boot, seed),
        "empty_rate": mean_ci(col("empty"), n_boot, seed),
        "abstain_rate": mean_ci(col("abstained"), n_boot, seed),
    }
    tp = [float(r["abstained"] and not r["answerable"]) for r in rows]
    pred = [float(r["abstained"]) for r in rows]
    pos = [float(not r["answerable"]) for r in rows]
    metrics["abstention_precision"] = ratio_ci(tp, pred, n_boot, seed)
    metrics["abstention_recall"] = ratio_ci(tp, pos, n_boot, seed)
    metrics["invalid_citation_share"] = ratio_ci(
        [float(r["n_invalid_cites"]) for r in rows], [float(r["n_cites"]) for r in rows],
        n_boot, seed)
    metrics["mean_reward"] = {name: mean_ci(col(name), n_boot, seed) for name in REWARD_FNS}
    words = [r["words"] for r in rows]
    metrics["length_words"] = {
        "mean": float(np.mean(words)) if words else None,
        "median": float(median(words)) if words else None,
        "p95": float(np.percentile(words, 95)) if words else None,
        "max": int(max(words)) if words else None,
    }
    if token_counts is not None:
        metrics["length_tokens"] = {
            "mean": float(np.mean(token_counts)), "p95": float(np.percentile(token_counts, 95)),
            "max": int(max(token_counts)),
        }
    by_type: dict[str, dict] = {}
    for qtype in sorted({r["qtype"] for r in rows}):
        sel = [r for r in rows if r["qtype"] == qtype]
        by_type[qtype] = {
            "n": len(sel),
            "mean_v2": float(np.mean([r["v2"] for r in sel])),
            "grounded_correct_rate": (float(np.mean([r["grounded_correct"] for r in sel]))
                                      if sel[0]["answerable"] else None),
        }
    return {
        "n": len(rows), "n_answerable": sum(answerable),
        "n_unanswerable": len(rows) - sum(answerable),
        "metrics": metrics, "by_qtype": by_type, "bootstrap": n_boot,
    }


def evaluate_policy(policy_name: str, examples: Sequence[Example], n_boot: int = 2000,
                    seed: int = 0) -> dict:
    """Evaluate one scripted policy from :mod:`groundrl.audit` with the same pipeline."""
    from .audit import POLICY_BY_NAME, run_policy

    spec = POLICY_BY_NAME[policy_name]
    completions = [run_policy(spec, e, seed) for e in examples]
    out = evaluate_completions(examples, completions, n_boot, seed)
    out["policy"] = policy_name
    return out


BASELINE_POLICIES = ("oracle", "extractive", "eip_id_shortcut", "refuse_all")


def run_baselines(
    data_dir: Path, split: str, out_dir: Path, policy_names: Sequence[str] = BASELINE_POLICIES,
    n_boot: int = 2000, seed: int = 0, limit: int | None = None,
) -> dict:
    """Evaluate scripted baseline policies on a split and write ``baseline_<split>.json``.

    These are reference points, not trained models: a trained model has to beat ``extractive``
    to show it learned more than locate-and-copy.
    """
    from .dataset import load_dataset

    ds = load_dataset(data_dir)
    examples = make_examples(ds.tasks[split][:limit], ds.index)
    result = {
        "kind": "baseline_eval", "mock": False, "split": split, "seed": seed,
        "note": "scripted policies, no model and no training; reference points only",
        "command": f"groundrl baseline-eval --split {split}",
        "policies": {n: evaluate_policy(n, examples, n_boot, seed) for n in policy_names},
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"baseline_{split}.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result
