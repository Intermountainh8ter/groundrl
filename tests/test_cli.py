from __future__ import annotations

import json

from groundrl import cli
from tests.helpers import make_docs


def test_build_tasks_then_check_data_on_a_temp_corpus(tmp_path, capsys):
    raw = tmp_path / "raw"
    raw.mkdir()
    for doc, text in make_docs(range(900, 930)).items():
        (raw / f"{doc}.md").write_text(text)
    assert cli.main(["build-tasks", "--data", str(tmp_path), "--seed", "5", "--k", "5"]) == 0
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["documents"] == 30 and out["total"] > 50
    # no manifest was written for the synthetic corpus, so check-data reports that
    try:
        cli.main(["check-data", "--data", str(tmp_path)])
    except FileNotFoundError:
        pass


def test_check_data_passes_on_the_committed_dataset(capsys):
    assert cli.main(["check-data", "--data", "data"]) == 0
    assert "data OK" in capsys.readouterr().out


def test_audit_subcommand_writes_files(tmp_path, capsys):
    out = tmp_path / "res"
    rc = cli.main(["audit", "--split", "val", "--limit", "8", "--boot", "10", "--out", str(out),
                   "--policies", "oracle", "empty"])
    assert rc == 0 and (out / "audit_val.json").exists()
    assert "Reward-hacking audit" in capsys.readouterr().out
