#!/usr/bin/env python
"""Score scripted adversarial policies under reward v0, v1, v2 (and v2 ablations).

No model and no GPU: every policy is a plain function. Writes ``results/audit_<split>.json`` and
``results/audit_<split>.md`` and prints the markdown table. Regenerate the README table with

    python scripts/reward_audit.py --split test
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from groundrl.audit import run_and_write  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=ROOT / "data")
    ap.add_argument("--split", choices=("train", "val", "test"), default="test")
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--boot", type=int, default=1000, help="bootstrap resamples for the CIs")
    ap.add_argument("--limit", type=int, default=None, help="only the first N tasks (smoke test)")
    ap.add_argument("--policies", nargs="*", help="subset of policy names")
    args = ap.parse_args(argv)
    _, table = run_and_write(args.data, args.split, args.out, args.seed, args.boot, args.limit,
                             args.policies)
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
