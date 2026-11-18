#!/usr/bin/env python
"""Simulated GRPO, extensional vs isomorphic verifier with symmetric grader noise, SIMULATED.

Each grader decision is flipped with probability n, so an arm with base accept a has effective
accept a*(1-2n)+n. Honest accept is 0.3 before noise; base exploit accept is 0.6 (extensional)
or 0.18 (isomorphic, assumed memoriser survival 0.3).

    python3 scripts/simulated_grpo_isomorphic_noisy.py --seed 7 \
        --out results/400-isomorphic-noisy-grader.json
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

NOISES = (0.0, 0.1, 0.2, 0.3, 0.5)


def noisy(accept: float, n: float) -> float:
    return accept * (1 - 2 * n) + n


def key(n: float) -> str:
    return "noise_" + f"{n:g}".replace(".", "p")


def run_arms(seed: int, *, tasks: int = 200, group: int = 8, steps: int = 200, lr: float = 0.5,
             p0: float = 0.05, honest_accept: float = 0.3, exploit_accept: float = 0.6,
             survival: float = 0.3, noises: tuple = NOISES) -> dict:
    base = {"extensional_noisy": exploit_accept, "isomorphic_noisy": exploit_accept * survival}
    arms: dict = {}
    for name, ea in base.items():
        arms[name] = {}
        for n in noises:
            rng = random.Random(f"{seed}:{name}:{n}")
            finals = [run_task(rng, 0.0, group=group, steps=steps, lr=lr, p0=p0,
                               honest_accept=noisy(honest_accept, n),
                               exploit_accept=noisy(ea, n))
                      for _ in range(tasks)]
            arms[name][key(n)] = {"final_exploit_prob": sum(finals) / len(finals),
                                  "noise": n, "effective_exploit_accept": noisy(ea, n),
                                  "effective_honest_accept": noisy(honest_accept, n)}
    return {"arms": arms,
            "config": {"tasks": tasks, "group": group, "steps": steps, "lr": lr, "p0": p0,
                       "honest_accept": honest_accept, "exploit_accept": exploit_accept,
                       "memoriser_survival_assumed": survival, "noises": list(noises)}}


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
        "command": "python3 scripts/simulated_grpo_isomorphic_noisy.py "
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
