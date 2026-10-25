#!/usr/bin/env python
"""Score a SIMULATED exploiting policy against four grader strictness rungs (no model, CPU only).

    python3 scripts/strictness_ladder.py --split test --seed 7 \
        --out results/50-strictness-ladder.json
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from groundrl.dataset import load_dataset  # noqa: E402
from groundrl.ladder import run_ladder  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=ROOT / "data")
    ap.add_argument("--split", choices=("train", "val", "test"), default="test")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    t0 = time.monotonic()
    ds = load_dataset(args.data)
    res = run_ladder(ds.tasks[args.split], ds.index)
    res["split"] = args.split
    res["simulated"] = True
    res["provenance"] = {
        "started_at": started,
        "seeds": [args.seed],
        "command": "python3 scripts/strictness_ladder.py " + " ".join(argv if argv is not None
                                                                     else sys.argv[1:]),
        "hardware": f"CPU only, {platform.machine()}, python {platform.python_version()}",
        "wall_s": round(time.monotonic() - t0, 3),
        "data_sources": [{"url": "https://raw.githubusercontent.com/ethereum/EIPs/master/EIPS/",
                          "license": "CC0-1.0"}],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("exploit_success_rate", "zero_spread_share")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
