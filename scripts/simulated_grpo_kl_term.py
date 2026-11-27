#!/usr/bin/env python
"""Simulated GRPO with a KL-style pull toward the reference logit, SIMULATED (no model).

The pull is applied on every step, including groups whose advantage is zero. Reports mean
final exploit probability per pull coefficient at exploit accept 0.6, honest accept 0.6.

    PYTHONPATH=src python3 scripts/simulated_grpo_kl_term.py --seed 7 \
        --out results/667-kl-term.json
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

BETAS = (0.0, 0.02, 0.1)
EXPLOIT_ACCEPT = 0.6
HONEST_ACCEPT = 0.6
P0 = 0.05
MIN_DROP = 0.05


def _key(b: float) -> str:
    return "b" + f"{b:.2f}".replace(".", "p")


def run_task(rng: random.Random, beta: float, *, group: int, steps: int, lr: float, p0: float,
             honest_accept: float, exploit_accept: float) -> float:
    """Return the final exploit probability for one task."""
    theta_ref = math.log(p0 / (1.0 - p0))
    theta = theta_ref
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
        grad = 0.0
        if std != 0.0:
            grad = sum(((r - mean) / (std + 1e-6)) * ((1.0 if a else 0.0) - p)
                       for a, r in zip(acts, rewards, strict=True)) / group
        theta += lr * (grad - beta * (theta - theta_ref))
        theta = max(-30.0, min(30.0, theta))
    return 1.0 / (1.0 + math.exp(-theta))


def run_arm(seed: int, *, betas=BETAS, tasks: int = 200, group: int = 8, steps: int = 200,
            lr: float = 0.5, p0: float = P0, honest_accept: float = HONEST_ACCEPT,
            exploit_accept: float = EXPLOIT_ACCEPT) -> dict:
    final: dict[str, float] = {}
    for b in betas:
        rng = random.Random(f"{seed}:{b}")
        finals = [run_task(rng, b, group=group, steps=steps, lr=lr, p0=p0,
                           honest_accept=honest_accept, exploit_accept=exploit_accept)
                  for _ in range(tasks)]
        final[_key(b)] = sum(finals) / len(finals)
    drop = {k: final[_key(0.0)] - v for k, v in final.items()}
    out = {
        "final_exploit_prob": final,
        "drop_vs_b0p00": drop,
        "config": {"tasks": tasks, "group": group, "steps": steps, "lr": lr, "p0": p0,
                   "honest_accept": honest_accept, "exploit_accept": exploit_accept,
                   "betas": list(betas)},
    }
    if "b0p10" in final and "b0p02" in final:
        out["drop_at_least_min"] = drop["b0p10"] >= MIN_DROP
        out["drop_0p10_not_smaller_than_0p02"] = drop["b0p10"] >= drop["b0p02"]
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
        "command": "python3 scripts/simulated_grpo_kl_term.py "
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
