#!/usr/bin/env python
"""Simulated GRPO strict-verifier arm, SIMULATED (no model, no corpus).

Exploit accept rates 0.0, 0.1, 0.2 against honest accept rates 0.5 .. 1.0 (step 0.1);
reports mean final exploit probability against the initial p0. No audit.

    PYTHONPATH=src python3 scripts/simulated_grpo_strict_arm.py --seed 7 \
        --out results/411-strict-arm.json
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
from simulated_grpo_exploit_accept_sweep import run_sweep  # noqa: E402

EXPLOIT = (0.0, 0.1, 0.2)
HONEST = tuple(round(0.5 + 0.1 * i, 1) for i in range(6))  # 0.5 .. 1.0
P0 = 0.05


def run_arm(seed: int, **kw) -> dict:
    res = run_sweep(seed, exploit=EXPLOIT, honest=HONEST, p0=P0, **kw)
    strict = res["exploit_prob_final"]["e0p00"]
    res["strict_stays_below_p0"] = all(v < P0 for v in strict.values())
    res["e0p20_h0p50_at_least_p0"] = res["exploit_prob_final"]["e0p20"]["h0p50"] >= P0
    return res


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
        "command": "python3 scripts/simulated_grpo_strict_arm.py "
        + " ".join(argv if argv is not None else sys.argv[1:]),
        "hardware": f"CPU only, {platform.machine()}, python {platform.python_version()}",
        "wall_s": round(time.monotonic() - t0, 3),
        "data_sources": [],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(res["exploit_prob_final"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
