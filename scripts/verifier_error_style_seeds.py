#!/usr/bin/env python
"""Multi-seed SIMULATED style-keyed versus independent verifier false-positive error (no model).

    python3 scripts/verifier_error_style_seeds.py --split test --group-size 8 \
        --groups-per-task 20 --error-rate 0.2 --seeds 7 13 21 \
        --out results/333-style-seeds.json
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from groundrl.dataset import load_dataset  # noqa: E402
from groundrl.verifier_error_style import run_verifier_error_style  # noqa: E402


def summarise(per_seed: dict[str, dict]) -> dict:
    raised = {s: r["raised_diff"] for s, r in per_seed.items()}
    base = {s: r["base_diff"] for s, r in per_seed.items()}
    rv = list(raised.values())
    return {
        "exploit_share_style_minus_independent_per_seed": raised,
        "exploit_share_style_minus_independent_min": min(rv),
        "exploit_share_style_minus_independent_mean": sum(rv) / len(rv),
        "base_rate_diff_per_seed": base,
        "base_rate_diff_max_abs": max(abs(v) for v in base.values()),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=ROOT / "data")
    ap.add_argument("--split", choices=("train", "val", "test"), default="test")
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--groups-per-task", type=int, default=20)
    ap.add_argument("--error-rate", type=float, default=0.2)
    ap.add_argument("--prob", type=float, default=0.5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[7, 13, 21])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    t0 = time.monotonic()
    ds = load_dataset(args.data)
    per_seed = {str(s): run_verifier_error_style(
        ds.tasks[args.split], ds.index, group_size=args.group_size,
        groups_per_task=args.groups_per_task, error_rate=args.error_rate,
        prob=args.prob, seed=s) for s in args.seeds}
    res = {
        "style_dependent_error": summarise(per_seed),
        "per_seed": per_seed,
        "split": args.split,
        "simulated": True,
        "provenance": {
            "started_at": started,
            "seeds": args.seeds,
            "command": "python3 scripts/verifier_error_style_seeds.py " + " ".join(
                argv if argv is not None else sys.argv[1:]),
            "hardware": f"CPU only, {platform.machine()}, python {platform.python_version()}",
            "wall_s": round(time.monotonic() - t0, 3),
            "data_sources": [{"url": "https://raw.githubusercontent.com/ethereum/EIPs/master/EIPS/",
                              "license": "CC0-1.0"}],
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(res["style_dependent_error"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
