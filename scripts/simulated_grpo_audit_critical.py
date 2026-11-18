#!/usr/bin/env python
"""Fine-grid audit-budget sweep around the break-even budget, SIMULATED (no model, no corpus).

Reuses ``run_task`` from simulated_grpo_audit_budget.py and reports the smallest budget on the
grid at which mean final exploit probability falls below 0.25.

    python3 scripts/simulated_grpo_audit_critical.py --seed 7 \
        --out results/397-audit-budget-critical.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_p = Path(__file__).resolve().parent / "simulated_grpo_audit_budget.py"
_spec = importlib.util.spec_from_file_location("sgab_base", _p)
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)

BUDGETS = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70)
THRESHOLD = 0.25


def run_critical(seed: int, *, budgets=BUDGETS, tasks: int = 200, group: int = 8,
                 steps: int = 200, lr: float = 0.5, p0: float = 0.05,
                 honest_accept: float = 0.3, exploit_accept: float = 0.6) -> dict:
    sweep: dict[str, float] = {}
    for b in budgets:
        rng = random.Random(f"{seed}:{b}")
        finals = [_base.run_task(rng, b, group=group, steps=steps, lr=lr, p0=p0,
                                 honest_accept=honest_accept, exploit_accept=exploit_accept)
                  for _ in range(tasks)]
        sweep["exploit_prob_final_" + _base._key(b)] = sum(finals) / len(finals)
    below = [b for b in budgets if sweep["exploit_prob_final_" + _base._key(b)] < THRESHOLD]
    return {
        "sweep": sweep,
        "critical": {
            "min_budget_below_0p25": min(below) if below else None,
            "break_even_budget": 1.0 - honest_accept / exploit_accept,
        },
        "config": {"tasks": tasks, "group": group, "steps": steps, "lr": lr, "p0": p0,
                   "honest_accept": honest_accept, "exploit_accept": exploit_accept,
                   "budgets": list(budgets), "threshold": THRESHOLD},
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    t0 = time.monotonic()
    res = run_critical(args.seed)
    res["simulated"] = True
    res["provenance"] = {
        "started_at": started,
        "seeds": [args.seed],
        "command": "python3 scripts/simulated_grpo_audit_critical.py "
        + " ".join(argv if argv is not None else sys.argv[1:]),
        "hardware": f"CPU only, {platform.machine()}, python {platform.python_version()}",
        "wall_s": round(time.monotonic() - t0, 3),
        "data_sources": [],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**res["sweep"], **res["critical"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
