#!/usr/bin/env python
"""Simulated GRPO, extensional versus isomorphic verifier arm, SIMULATED (no model, no corpus).

The isomorphic arm re-checks each accepted rollout on a perturbed instance, so the exploit
accept probability is scaled by an ASSUMED memoriser survival rate. Honest accept is unchanged.

    python3 scripts/simulated_grpo_isomorphic_arm.py --seed 7 \
        --out results/399-simulated-grpo-isomorphic-arm.json
"""
from __future__ import annotations

import argparse
import json
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from simulated_grpo_audit_budget import run_task  # noqa: E402


def run_arms(seed: int, *, tasks: int = 200, group: int = 8, steps: int = 200, lr: float = 0.5,
             p0: float = 0.05, honest_accept: float = 0.3, exploit_accept: float = 0.6,
             survival: float = 0.3) -> dict:
    accepts = {"extensional": exploit_accept, "isomorphic": exploit_accept * survival}
    arms = {}
    for name, ea in accepts.items():
        rng = random.Random(f"{seed}:{name}")
        finals = [run_task(rng, 0.0, group=group, steps=steps, lr=lr, p0=p0,
                           honest_accept=honest_accept, exploit_accept=ea)
                  for _ in range(tasks)]
        arms[name] = {"final_exploit_prob": sum(finals) / len(finals),
                      "exploit_accept": ea}
    return {"arms": arms,
            "config": {"tasks": tasks, "group": group, "steps": steps, "lr": lr, "p0": p0,
                       "honest_accept": honest_accept, "exploit_accept": exploit_accept,
                       "memoriser_survival_assumed": survival}}


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
        "command": "python3 scripts/simulated_grpo_isomorphic_arm.py "
        + " ".join(argv if argv is not None else sys.argv[1:]),
        "hardware": f"CPU only, {platform.machine()}, python {platform.python_version()}",
        "wall_s": round(time.monotonic() - t0, 3),
        "data_sources": [],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(res["arms"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
