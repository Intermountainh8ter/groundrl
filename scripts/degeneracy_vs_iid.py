#!/usr/bin/env python
"""Pooled i.i.d. versus per-task zero-variance group prediction (no model, CPU only).

    python3 scripts/degeneracy_vs_iid.py --split test --group-size 8 --groups-per-task 20 \
        --seed 7 --out results/234-degeneracy-vs-iid.json
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
from groundrl.degeneracy_vs_iid import run_degeneracy_vs_iid  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=ROOT / "data")
    ap.add_argument("--split", choices=("train", "val", "test"), default="test")
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--groups-per-task", type=int, default=20)
    ap.add_argument("--probs", type=float, nargs="+", default=[0.1, 0.5, 0.9])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    t0 = time.monotonic()
    ds = load_dataset(args.data)
    res = run_degeneracy_vs_iid(ds.tasks[args.split], ds.index, group_size=args.group_size,
                               groups_per_task=args.groups_per_task, probs=args.probs,
                               seed=args.seed)
    res["split"] = args.split
    res["simulated"] = True
    res["provenance"] = {
        "started_at": started,
        "seeds": [args.seed],
        "command": "python3 scripts/degeneracy_vs_iid.py " + " ".join(argv if argv is not None
                                                                     else sys.argv[1:]),
        "hardware": f"CPU only, {platform.machine()}, python {platform.python_version()}",
        "wall_s": round(time.monotonic() - t0, 3),
        "data_sources": [{"url": "https://raw.githubusercontent.com/ethereum/EIPs/master/EIPS/",
                          "license": "CC0-1.0"}],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(res["degeneracy"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
