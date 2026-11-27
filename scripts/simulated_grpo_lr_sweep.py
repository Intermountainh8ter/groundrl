#!/usr/bin/env python
"""Simulated GRPO learning-rate sweep at a loose grader, SIMULATED (no model).

Reports mean final exploit probability per learning rate at exploit accept 0.9, honest
accept 0.6, no KL term, no audit.

    PYTHONPATH=src python3 scripts/simulated_grpo_lr_sweep.py --seed 7 \
        --out results/676-lr-sweep.json
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

LRS = (0.02, 0.05, 0.1, 0.2, 0.5)
EXPLOIT_ACCEPT = 0.9
HONEST_ACCEPT = 0.6
P0 = 0.05
LOW_LR = 0.1
HIGH_LR = 0.5
LOW_MAX = 0.3
HIGH_MIN = 0.5


def _key(lr: float) -> str:
    return "lr" + f"{lr:.2f}".replace(".", "p")


def run_task(rng: random.Random, *, group: int, steps: int, lr: float, p0: float,
             honest_accept: float, exploit_accept: float) -> float:
    """Return the final exploit probability for one task."""
    theta = math.log(p0 / (1.0 - p0))
    for _ in range(steps):
        p = 1.0 / (1.0 + math.exp(-theta))
        acts, rewards = [], []
        for _ in range(group):
            exploit = rng.random() < p
            accept = exploit_accept if exploit else honest_accept
            acts.append(exploit)
            rewards.append(1.0 if rng.random() < accept else 0.0)
        mean = sum(rewards) / group
        std = math.sqrt(sum((r - mean) ** 2 for r in rewards) / group)
        if std != 0.0:
            grad = sum(((r - mean) / (std + 1e-6)) * ((1.0 if a else 0.0) - p)
                       for a, r in zip(acts, rewards, strict=True)) / group
            theta += lr * grad
            theta = max(-30.0, min(30.0, theta))
    return 1.0 / (1.0 + math.exp(-theta))


def run_arm(seed: int, *, lrs=LRS, tasks: int = 200, group: int = 8, steps: int = 200,
            p0: float = P0, honest_accept: float = HONEST_ACCEPT,
            exploit_accept: float = EXPLOIT_ACCEPT) -> dict:
    final: dict[str, float] = {}
    for lr in lrs:
        rng = random.Random(f"{seed}:{lr}")
        finals = [run_task(rng, group=group, steps=steps, lr=lr, p0=p0,
                           honest_accept=honest_accept, exploit_accept=exploit_accept)
                  for _ in range(tasks)]
        final[_key(lr)] = sum(finals) / len(finals)
    out = {
        "final_exploit_prob": final,
        "config": {"tasks": tasks, "group": group, "steps": steps, "p0": p0,
                   "honest_accept": honest_accept, "exploit_accept": exploit_accept,
                   "lrs": list(lrs)},
    }
    lo, hi = _key(LOW_LR), _key(HIGH_LR)
    if lo in final and hi in final:
        out["low_below_max"] = final[lo] < LOW_MAX
        out["high_at_least_min"] = final[hi] >= HIGH_MIN
        out["hypothesis_supported"] = out["low_below_max"] and out["high_at_least_min"]
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    t0 = time.monotonic()
    res = run_arm(args.seed)
    res["simulated"] = True
    res["provenance"] = {
        "started_at": started,
        "seeds": [args.seed],
        "command": "python3 scripts/simulated_grpo_lr_sweep.py "
        + " ".join(argv if argv is not None else sys.argv[1:]),
        "hardware": f"CPU only, {platform.machine()}, python {platform.python_version()}",
        "wall_s": round(time.monotonic() - t0, 3),
        "data_sources": [],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(res["final_exploit_prob"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
