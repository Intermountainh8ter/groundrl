#!/usr/bin/env python
"""Simulated GRPO audit-budget sweep, SIMULATED (no model, no corpus).

A two-action softmax policy per task (honest versus exploit) is trained with
group-normalised advantages. A cheap check accepts honest rollouts with probability
``honest_accept`` and exploit rollouts with ``exploit_accept``; a strong verifier re-checks a
fraction ``budget`` of exploit-accepted rollouts and rejects them.

    python3 scripts/simulated_grpo_audit_budget.py --seed 7 \
        --out results/394-simulated-grpo-audit-budget.json
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import random
import time
from datetime import datetime, timezone
from pathlib import Path

BUDGETS = (0.0, 0.25, 0.5, 0.75, 1.0)


def _key(b: float) -> str:
    return "b" + f"{b:g}".replace(".", "p")


def run_task(rng: random.Random, budget: float, *, group: int, steps: int, lr: float,
             p0: float, honest_accept: float, exploit_accept: float) -> float:
    """Return the final exploit probability for one task."""
    theta = math.log(p0 / (1.0 - p0))
    for _ in range(steps):
        p = 1.0 / (1.0 + math.exp(-theta))
        acts, rewards = [], []
        for _ in range(group):
            exploit = rng.random() < p
            if exploit:
                r = 1.0 if rng.random() < exploit_accept else 0.0
                if r and rng.random() < budget:
                    r = 0.0
            else:
                r = 1.0 if rng.random() < honest_accept else 0.0
            acts.append(exploit)
            rewards.append(r)
        mean = sum(rewards) / group
        std = math.sqrt(sum((r - mean) ** 2 for r in rewards) / group)
        if std == 0.0:
            continue
        grad = sum(((r - mean) / (std + 1e-6)) * ((1.0 if a else 0.0) - p)
                   for a, r in zip(acts, rewards, strict=True)) / group
        theta += lr * grad
        theta = max(-30.0, min(30.0, theta))
    return 1.0 / (1.0 + math.exp(-theta))


def run_sweep(seed: int, *, budgets=BUDGETS, tasks: int = 200, group: int = 8, steps: int = 200,
              lr: float = 0.5, p0: float = 0.05, honest_accept: float = 0.3,
              exploit_accept: float = 0.6) -> dict:
    sweep: dict[str, float] = {}
    for b in budgets:
        rng = random.Random(f"{seed}:{b}")
        finals = [run_task(rng, b, group=group, steps=steps, lr=lr, p0=p0,
                           honest_accept=honest_accept, exploit_accept=exploit_accept)
                  for _ in range(tasks)]
        sweep["exploit_prob_final_" + _key(b)] = sum(finals) / len(finals)
    vals = [sweep["exploit_prob_final_" + _key(b)] for b in budgets]
    return {
        "sweep": sweep,
        "non_increasing_in_budget": all(x >= y for x, y in zip(vals, vals[1:], strict=False)),
        "effective_exploit_accept": {_key(b): exploit_accept * (1.0 - b) for b in budgets},
        "config": {"tasks": tasks, "group": group, "steps": steps, "lr": lr, "p0": p0,
                   "honest_accept": honest_accept, "exploit_accept": exploit_accept,
                   "budgets": list(budgets)},
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    t0 = time.monotonic()
    res = run_sweep(args.seed)
    res["simulated"] = True
    res["provenance"] = {
        "started_at": started,
        "seeds": [args.seed],
        "command": "python3 scripts/simulated_grpo_audit_budget.py "
        + " ".join(argv if argv is not None else __import__("sys").argv[1:]),
        "hardware": f"CPU only, {platform.machine()}, python {platform.python_version()}",
        "wall_s": round(time.monotonic() - t0, 3),
        "data_sources": [],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(res["sweep"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
