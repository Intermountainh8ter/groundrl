"""Command-line control for ``train/sft.py``, ``train/grpo.py`` and ``train/eval.py``.

Every stage supports ``--dry-run`` (validate the config, load the dataset, estimate tokens with a
stub tokenizer and print the plan; no torch, no download) and ``--mock`` (run the real control
flow against the fake stack from :mod:`groundrl.mockstack`; nothing is trained and the output is
labelled MOCK). Without either flag a stage imports the training stack and runs for real.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .trainconf import ConfigError, apply_overrides, cli_overrides, load_config
from .trainplan import build_plan, format_plan

STAGES = ("sft", "grpo", "eval")
MOCK_BANNER = ("*** MOCK RUN: nothing is trained and no real model is used. Output exists only "
               "to test the plumbing and is never counted as a result. ***")


def build_parser(stage: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog=f"train/{stage}.py", description=__doc__)
    ap.add_argument("--config", required=True, type=Path, help="JSON config (train/configs)")
    ap.add_argument("--out", type=Path, default=None, help="output directory")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit; no torch")
    ap.add_argument("--mock", action="store_true", help="plumbing test with a fake stack (MOCK)")
    ap.add_argument("--repo-root", type=Path, default=Path.cwd())
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None, help="learning rate for this stage")
    ap.add_argument("--max-len", type=int, default=None, help="total token budget (prompt+answer)")
    ap.add_argument("--set", dest="sets", action="append", default=[], metavar="KEY=JSON",
                    help="override any config value by dotted path, e.g. grpo.args.beta=0.04")
    ap.add_argument("--limit", type=int, default=None, help="use only the first N examples")
    if stage == "grpo":
        ap.add_argument("--group-size", type=int, default=None, help="completions per prompt")
        ap.add_argument("--kl-beta", type=float, default=None, help="KL coefficient")
        ap.add_argument("--init-from", default=None, help="start from this model (merged SFT)")
    if stage == "sft":
        ap.add_argument("--no-merge", action="store_true", help="skip merging the adapter")
    if stage == "eval":
        ap.add_argument("--model", default=None, help="model path (default: config base model)")
        ap.add_argument("--adapter", default=None, help="LoRA adapter directory to apply")
        ap.add_argument("--split", default=None, help="split to evaluate (default: eval.split)")
        ap.add_argument("--label", default=None, help="name used in the result file")
    return ap


def _overrides(args: argparse.Namespace, cfg: dict[str, Any], stage: str) -> dict[str, Any]:
    ov = cli_overrides(
        cfg, seed=args.seed, lr=args.lr, group_size=getattr(args, "group_size", None),
        kl_beta=getattr(args, "kl_beta", None), max_len=args.max_len, sets=args.sets,
        stage=stage if stage != "eval" else "grpo")
    if getattr(args, "init_from", None):
        ov["model.init_from"] = args.init_from
    if args.limit is not None and stage != "eval":
        ov["data.limit_train"] = args.limit
        ov["data.limit_eval"] = max(1, args.limit // 4)
    return ov


def _mock_stack(cfg: dict[str, Any], repo_root: Path) -> SimpleNamespace:
    """A fake stack whose model answers with the scripted ``extractive`` policy."""
    from .audit import POLICY_BY_NAME, run_policy
    from .dataset import load_dataset
    from .example import make_examples
    from .mockstack import make_mock_stack
    from .prompts import build_messages, render_chat_stub

    data_dir = Path(cfg["data"]["data_dir"])
    ds = load_dataset(data_dir if data_dir.is_absolute() else repo_root / data_dir)
    lookup = {}
    for split in ("train", "val", "test"):
        for ex in make_examples(ds.tasks[split], ds.index):
            prompt = render_chat_stub(build_messages(ex), True, cfg["data"]["enable_thinking"])
            lookup[prompt] = ex
    spec = POLICY_BY_NAME["extractive"]

    def policy(prompt: str) -> str:
        ex = lookup.get(prompt)
        return run_policy(spec, ex) if ex is not None else ""

    return make_mock_stack(policy)


def main(
    stage: str, argv: Sequence[str] | None = None, stack: SimpleNamespace | None = None,
) -> int:
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}")
    args = build_parser(stage).parse_args(argv)
    try:
        cfg = load_config(args.config)
        cfg = apply_overrides(cfg, _overrides(args, cfg, stage))
    except (ConfigError, OSError) as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    if args.dry_run:
        plan = build_plan(cfg, stage, repo_root=args.repo_root)
        print(format_plan(plan))
        return 0
    mock = args.mock
    if mock:
        print(MOCK_BANNER)
        stack = _mock_stack(cfg, args.repo_root)
    base_out = args.out or Path("outputs") / ("mock" if mock else "") / cfg["name"] / stage
    from . import trl_backend

    if stage == "sft":
        summary = trl_backend.run_sft(cfg, base_out, stack, mock, args.repo_root,
                                      merge=not args.no_merge)
    elif stage == "grpo":
        summary = trl_backend.run_grpo(cfg, base_out, stack, mock, args.repo_root)
    else:
        label = args.label or f"{cfg['name']}-{'adapter' if args.adapter else 'model'}"
        results_dir = args.out or (Path("outputs/mock/results") if mock else Path("results"))
        out_path = Path(results_dir) / f"eval_{label}.json"
        summary = trl_backend.run_eval(cfg, args.model, args.adapter, out_path, args.split, stack,
                                       mock, args.limit, args.repo_root, label)
        print(f"wrote {out_path}")
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("log_history", "samples", "metrics", "by_qtype")}, indent=2,
                     default=str))
    return 0
