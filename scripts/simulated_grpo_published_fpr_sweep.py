#!/usr/bin/env python
"""Simulated GRPO honest-accept crossover at published buggy-verifier FPRs, SIMULATED.

Exploit accept rates 0.832 and 0.869 (the false-positive rates quoted in arXiv 2606.01066);
honest accept grid 0.50 to 1.00. No model, no corpus, no audit.

    PYTHONPATH=src python3 scripts/simulated_grpo_published_fpr_sweep.py --seed 7 \
        --out results/409-published-fpr-sweep.json
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

EXPLOIT = (0.832, 0.869)
HONEST = tuple(round(0.50 + 0.05 * i, 2) for i in range(11))  # 0.50 .. 1.00


def run(seed: int, *, exploit=EXPLOIT, honest=HONEST, **kw) -> dict:
    res = run_sweep(seed, exploit=exploit, honest=honest, **kw)
    cross = res["crossover_by_exploit_accept"]
    keys = [k for k in cross]
    a, b = (cross[keys[0]], cross[keys[1]]) if len(keys) == 2 else (None, None)
    res["crossover_not_lower_at_higher_exploit_accept"] = (
        None if a is None or b is None else b >= a)
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    t0 = time.monotonic()
    res = run(args.seed)
    res["simulated"] = True
    res["provenance"] = {
        "started_at": started,
        "seeds": [args.seed],
        "command": "python3 scripts/simulated_grpo_published_fpr_sweep.py "
        + " ".join(argv if argv is not None else sys.argv[1:]),
        "hardware": f"CPU only, {platform.machine()}, python {platform.python_version()}",
        "wall_s": round(time.monotonic() - t0, 3),
        "data_sources": [],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(res["crossover_by_exploit_accept"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
