#!/usr/bin/env python
"""Simulated GRPO initial-exploit-probability (p0) sweep at a loose grader, SIMULATED (no model).

Reports mean final exploit probability per initial exploit probability at exploit accept 0.9, honest
accept 0.6, no KL term, no audit.

    PYTHONPATH=src python3 scripts/simulated_grpo_p0_sweep.py --seed 7 \
        --out results/681-p0-sweep.json
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

P0S = (0.001, 0.01, 0.05, 0.1, 0.2)
EXPLOIT_ACCEPT = 0.9
HONEST_ACCEPT = 0.6
LR = 0.5
LOW_P0 = 0.001
HIGH_P0 = 0.2
LOW_MAX = 0.5
HIGH_MIN = 0.9


def _key(p0: float) -> str:
    return "p0_" + f"{p0:g}".replace(".", "p")


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


def run_arm(seed: int, *, p0s=P0S, tasks: int = 200, group: int = 8, steps: int = 200,
            lr: float = LR, honest_accept: float = HONEST_ACCEPT,
            exploit_accept: float = EXPLOIT_ACCEPT) -> dict:
    final: dict[str, float] = {}
    for p0 in p0s:
        rng = random.Random(f"{seed}:{p0}")
        finals = [run_task(rng, group=group, steps=steps, lr=lr, p0=p0,
                           honest_accept=honest_accept, exploit_accept=exploit_accept)
                  for _ in range(tasks)]
        final[_key(p0)] = sum(finals) / len(finals)
    out = {
        "final_exploit_prob": final,
        "config": {"tasks": tasks, "group": group, "steps": steps, "lr": lr,
                   "honest_accept": honest_accept, "exploit_accept": exploit_accept,
                   "p0s": list(p0s)},
    }
    lo, hi = _key(LOW_P0), _key(HIGH_P0)
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
        "command": "python3 scripts/simulated_grpo_p0_sweep.py "
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
