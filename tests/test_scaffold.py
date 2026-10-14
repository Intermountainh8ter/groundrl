from __future__ import annotations

import importlib
import sys

import groundrl
from groundrl import cli


def test_version_is_semver_like():
    parts = groundrl.__version__.split(".")
    assert len(parts) == 3 and all(p.isdigit() for p in parts)


def test_cli_without_command_prints_help(capsys):
    assert cli.main([]) == 0
    assert "groundrl" in capsys.readouterr().out


def test_cli_version_flag(capsys):
    try:
        cli.main(["--version"])
    except SystemExit as exc:
        assert exc.code == 0
    assert groundrl.__version__ in capsys.readouterr().out


def test_heavy_dependencies_not_imported_by_core():
    for name in ("groundrl", "groundrl.cli"):
        importlib.import_module(name)
    for heavy in ("torch", "transformers", "trl", "peft", "datasets", "vllm"):
        assert heavy not in sys.modules, f"{heavy} must stay an optional, lazily imported extra"
