from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import groundrl
from groundrl import cli

SRC = Path(__file__).resolve().parents[1] / "src"

def _imports_after(statements: str) -> list[str]:
    """Run `statements` in a fresh interpreter; return which heavy modules it imported.

    sys.modules is global to a pytest session, so asserting "torch is not in
    sys.modules" inside the session stops testing anything the moment any other
    test imports torch -- it then passes for the wrong reason. A child
    interpreter is the only place this property is actually observable.
    """
    source = "\n".join([
        "import sys",
        f"sys.path.insert(0, {str(SRC)!r})",
        textwrap.dedent(statements).strip(),
        'heavy = [m for m in ("torch", "transformers", "trl", "peft", "datasets", "vllm")',
        "         if m in sys.modules]",
        'print(",".join(heavy))',
    ])
    r = subprocess.run([sys.executable, "-c", source], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    return [m for m in r.stdout.strip().split(",") if m]


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
    imported = _imports_after("import groundrl\nimport groundrl.cli")
    assert imported == [], f"{imported} must stay optional, lazily imported extras"


def test_module_entry_point_runs(monkeypatch, capsys):
    import runpy

    monkeypatch.setattr(sys, "argv", ["groundrl"])
    try:
        runpy.run_module("groundrl", run_name="__main__")
    except SystemExit as exc:
        assert exc.code == 0
    assert "groundrl" in capsys.readouterr().out
