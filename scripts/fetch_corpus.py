#!/usr/bin/env python
"""Download the curated EIP corpus into data/raw and write data/MANIFEST.json (sha256 per file).

Only raw.githubusercontent.com is contacted. EIPs are CC0. Re-running overwrites files in place;
`git diff data/MANIFEST.json` then shows exactly what upstream changed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from groundrl.corpus import CURATED_EIPS, fetch_corpus, urllib_opener  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "raw")
    ap.add_argument("--only", type=int, nargs="*", help="fetch just these EIP numbers")
    args = ap.parse_args(argv)
    numbers = args.only or CURATED_EIPS
    manifest = fetch_corpus(numbers, args.out, urllib_opener())
    print(f"kept {len(manifest['documents'])} documents, skipped {len(manifest['skipped'])}")
    for item in manifest["skipped"]:
        print(f"  skipped EIP-{item['number']}: {item['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
