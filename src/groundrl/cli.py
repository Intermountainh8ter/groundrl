"""Command line entry point (``groundrl``). Subcommands are registered as phases land."""
from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from . import __version__

DEFAULT_DATA = Path("data")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="groundrl",
        description="Deterministic citation-grounding reward for RL on small models.",
    )
    parser.add_argument("--version", action="version", version=f"groundrl {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    bt = sub.add_parser("build-tasks", help="chunk the raw corpus and generate the task splits")
    bt.add_argument("--data", type=Path, default=DEFAULT_DATA)
    bt.add_argument("--seed", type=int, default=None)
    bt.add_argument("--k", type=int, default=None, help="chunks per context")

    cd = sub.add_parser("check-data", help="verify manifest hashes and task invariants")
    cd.add_argument("--data", type=Path, default=DEFAULT_DATA)

    au = sub.add_parser("audit", help="score scripted adversarial policies under each reward")
    au.add_argument("--data", type=Path, default=DEFAULT_DATA)
    au.add_argument("--split", choices=("train", "val", "test"), default="test")
    au.add_argument("--out", type=Path, default=Path("results"))
    au.add_argument("--seed", type=int, default=0)
    au.add_argument("--boot", type=int, default=1000)
    au.add_argument("--limit", type=int, default=None)
    au.add_argument("--policies", nargs="*")
    return parser


def _cmd_audit(args: argparse.Namespace) -> int:
    from .audit import run_and_write

    _, table = run_and_write(args.data, args.split, args.out, args.seed, args.boot, args.limit,
                             args.policies)
    print(table)
    return 0


def _cmd_build_tasks(args: argparse.Namespace) -> int:
    from .dataset import build_dataset
    from .tasks import GenConfig

    cfg = GenConfig()
    if args.seed is not None:
        cfg.seed = args.seed
    if args.k is not None:
        cfg.k = args.k
    summary = build_dataset(args.data, cfg)
    print(json.dumps({k: summary[k] for k in ("documents", "chunks", "facts", "tasks", "total")}))
    return 0


def _cmd_check_data(args: argparse.Namespace) -> int:
    from .corpus import verify_manifest
    from .dataset import check_dataset, load_dataset

    problems = verify_manifest(args.data) + check_dataset(load_dataset(args.data))
    for p in problems:
        print(f"PROBLEM: {p}")
    print("data OK" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    return COMMANDS[args.command](args)


COMMANDS = {"build-tasks": _cmd_build_tasks, "check-data": _cmd_check_data, "audit": _cmd_audit}
