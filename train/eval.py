#!/usr/bin/env python
"""Entry point for the eval stage; see `python train/eval.py --help` and docs/RUNBOOK.md."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from groundrl.trainctl import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main("eval"))
