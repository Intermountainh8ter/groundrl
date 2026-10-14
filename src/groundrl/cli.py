"""Command line entry point (``groundrl``). Subcommands are registered as phases land."""
from __future__ import annotations

import argparse
from collections.abc import Sequence

from . import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="groundrl",
        description="Deterministic citation-grounding reward for RL on small models.",
    )
    parser.add_argument("--version", action="version", version=f"groundrl {__version__}")
    parser.add_subparsers(dest="command", metavar="<command>")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    return 0  # pragma: no cover - no subcommands are registered yet
