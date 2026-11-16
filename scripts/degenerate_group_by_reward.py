#!/usr/bin/env python
"""Share of SIMULATED GRPO groups with all-identical rewards under reward v0, v1, v2 (no model).

    python3 scripts/degenerate_group_by_reward.py --split test --group-size 8 --groups-per-task 20 \
        --prob 0.5 --seed 7 --out results/282-degenerate-group-by-reward.json
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
from groundrl.degenerate_group_by_reward import run_degenerate_group_by_reward  # noqa: E402
from groundrl.example import make_examples  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=ROOT / "data")
    ap.add_argument("--split", choices=("train", "val", "test"), default="test")
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--groups-per-task", type=int, default=20)
    ap.add_argument("--prob", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    t0 = time.monotonic()
    ds = load_dataset(args.data)
    examples = make_examples(ds.tasks[args.split], ds.index)
    res = run_degenerate_group_by_reward(examples, group_size=args.group_size,
                                       groups_per_task=args.groups_per_task, prob=args.prob,
                                       seed=args.seed)
    res["split"] = args.split
    res["simulated"] = True
    res["provenance"] = {
        "started_at": started,
        "seeds": [args.seed],
        "command": "python3 scripts/degenerate_group_by_reward.py " + " ".join(
            argv if argv is not None else sys.argv[1:]),
        "hardware": f"CPU only, {platform.machine()}, python {platform.python_version()}",
        "wall_s": round(time.monotonic() - t0, 3),
        "data_sources": [{"url": "https://raw.githubusercontent.com/ethereum/EIPs/master/EIPS/",
                          "license": "CC0-1.0"}],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({v: r["mean_share"]
                      for v, r in res["degenerate_group_fraction_by_reward"].items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
