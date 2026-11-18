#!/usr/bin/env python
"""Simulated GRPO honest-accept sweep, SIMULATED (no model, no corpus).

Holds the exploit accept rate fixed and sweeps the honest accept rate, no audit.

    PYTHONPATH=src python3 scripts/simulated_grpo_honest_accept_sweep.py --seed 7 \
        --out results/405-honest-accept-sweep.json
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

HONEST = (0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)
THRESHOLD = 0.5


def _key(h: float) -> str:
    return "h" + f"{h:.2f}".replace(".", "p")


def run_sweep(seed: int, *, honest=HONEST, tasks: int = 200, group: int = 8, steps: int = 200,
              lr: float = 0.5, p0: float = 0.05, exploit_accept: float = 0.6) -> dict:
    sweep: dict[str, float] = {}
    for h in honest:
        rng = random.Random(f"{seed}:{h}")
        finals = [run_task(rng, 0.0, group=group, steps=steps, lr=lr, p0=p0,
                           honest_accept=h, exploit_accept=exploit_accept)
                  for _ in range(tasks)]
        sweep["exploit_prob_final_" + _key(h)] = sum(finals) / len(finals)
    vals = [sweep["exploit_prob_final_" + _key(h)] for h in honest]
    below = [h for h, v in zip(honest, vals, strict=True) if v < THRESHOLD]
    return {
        "sweep": sweep,
        "crossover": {"min_honest_accept_exploit_lt_0p5": min(below) if below else None,
                      "threshold": THRESHOLD},
        "non_increasing_in_honest_accept": all(
            x >= y for x, y in zip(vals, vals[1:], strict=False)),
        "config": {"tasks": tasks, "group": group, "steps": steps, "lr": lr,
                   "p0": p0, "exploit_accept": exploit_accept,
                   "honest_accepts": list(honest)},
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
        "command": "python3 scripts/simulated_grpo_honest_accept_sweep.py "
        + " ".join(argv if argv is not None else sys.argv[1:]),
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
