#!/usr/bin/env python
"""Seed spread of the honest-accept crossover at published FPRs 0.832, 0.869, SIMULATED (no model).

    PYTHONPATH=src python3 scripts/simulated_grpo_published_fpr_seeds.py --seeds 7 11 13 17 19 \
        --out results/410-published-fpr-seeds.json
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from simulated_grpo_exploit_accept_sweep import _key, run_sweep  # noqa: E402

EXPLOITS = (0.832, 0.869)
HONEST = tuple(round(0.6 + 0.01 * i, 2) for i in range(31))  # 0.60 .. 0.90


def spread(seeds, exploit, *, honest=HONEST, **kw) -> dict:
    per_seed: dict[str, float | None] = {}
    for s in seeds:
        r = run_sweep(s, exploit=(exploit,), honest=honest, **kw)
        per_seed[str(s)] = r["crossover_by_exploit_accept"][_key("e", exploit)]
    vals = [v for v in per_seed.values() if v is not None]
    complete = len(vals) == len(per_seed) and bool(vals)
    return {
        "per_seed": per_seed,
        "min": min(vals) if vals else None,
        "max": max(vals) if vals else None,
        "max_minus_min": round(max(vals) - min(vals), 6) if complete else None,
        "mean": round(sum(vals) / len(vals), 6) if complete else None,
        "all_seeds_have_crossover": complete,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=[7, 11, 13, 17, 19])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    t0 = time.monotonic()
    res = {"crossover_seed_spread": {_key("e", e): spread(args.seeds, e) for e in EXPLOITS},
           "simulated": True,
           "config": {"exploit_accepts": list(EXPLOITS), "honest_accepts": list(HONEST),
                      "tasks": 200, "group": 8, "steps": 200, "lr": 0.5, "p0": 0.05,
                      "threshold": 0.5}}
    res["provenance"] = {
        "started_at": started, "seeds": args.seeds,
        "command": "python3 scripts/simulated_grpo_published_fpr_seeds.py "
        + " ".join(argv if argv is not None else sys.argv[1:]),
        "hardware": f"CPU only, {platform.machine()}, python {platform.python_version()}",
        "wall_s": round(time.monotonic() - t0, 3), "data_sources": [],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(res["crossover_seed_spread"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
