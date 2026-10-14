#!/usr/bin/env python
"""Render results/*.json into results/REPORT.md. Only what is on disk is rendered."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from groundrl.report import render_report  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    ap.add_argument("--out", type=Path, default=None, help="default: <results>/REPORT.md")
    args = ap.parse_args(argv)
    text = render_report(args.results)
    out = args.out or args.results / "REPORT.md"
    out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
