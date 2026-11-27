#!/usr/bin/env python
"""Simulated GRPO audit with correlated strong-verifier misses, SIMULATED (no model, no corpus).

Same two-action softmax loop as ``simulated_grpo_audit_budget.py``. A strong verifier audits a
fraction ``budget`` of exploit-accepted rollouts. Arms differ in the probability that the audit
misses an audited exploit: none, a constant, or equal to the current exploit probability p.

    python3 scripts/simulated_grpo_correlated_audit.py --seed 7 \
        --out results/696-correlated-audit.json
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

ARMS = {"miss_none": ("const", 0.0), "miss_const_0p25": ("const", 0.25),
        "miss_prop_p": ("prop", 1.0)}


def _miss(kind: str, value: float, p: float) -> float:
    return value if kind == "const" else value * p


def run_task(rng: random.Random, kind: str, value: float, *, budget: float, group: int,
             steps: int, lr: float, p0: float, honest_accept: float,
             exploit_accept: float) -> float:
    """Return the final exploit probability for one task."""
    theta = math.log(p0 / (1.0 - p0))
    for _ in range(steps):
        p = 1.0 / (1.0 + math.exp(-theta))
        miss = _miss(kind, value, p)
        acts, rewards = [], []
        for _ in range(group):
            exploit = rng.random() < p
            if exploit:
                r = 1.0 if rng.random() < exploit_accept else 0.0
                if r and rng.random() < budget and rng.random() >= miss:
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


def run_arms(seed: int, *, tasks: int = 200, group: int = 8, steps: int = 200, lr: float = 0.5,
             p0: float = 0.05, honest_accept: float = 0.3, exploit_accept: float = 0.6,
             budget: float = 0.5) -> dict:
    arms: dict[str, dict] = {}
    for name, (kind, value) in ARMS.items():
        rng = random.Random(f"{seed}:{name}")
        finals = [run_task(rng, kind, value, budget=budget, group=group, steps=steps, lr=lr,
                           p0=p0, honest_accept=honest_accept, exploit_accept=exploit_accept)
                  for _ in range(tasks)]
        arms[name] = {"exploit_prob_final": sum(finals) / len(finals), "miss_kind": kind,
                      "miss_value": value}
    return {"arms": arms,
            "config": {"tasks": tasks, "group": group, "steps": steps, "lr": lr, "p0": p0,
                       "honest_accept": honest_accept, "exploit_accept": exploit_accept,
                       "budget": budget}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    t0 = time.monotonic()
    res = run_arms(args.seed)
    res["simulated"] = True
    res["provenance"] = {
        "started_at": started,
        "seeds": [args.seed],
        "command": "python3 scripts/simulated_grpo_correlated_audit.py "
        + " ".join(argv if argv is not None else sys.argv[1:]),
        "hardware": f"CPU only, {platform.machine()}, python {platform.python_version()}",
        "wall_s": round(time.monotonic() - t0, 3),
        "data_sources": [],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v["exploit_prob_final"] for k, v in res["arms"].items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
