import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "persistent_seeds", ROOT / "scripts" / "verifier_error_persistent_seeds.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_summarise_keys():
    per = {"7": {"zero_variance_diff": 0.3, "exploit_share_diff": 0.01},
           "13": {"zero_variance_diff": 0.25, "exploit_share_diff": -0.02}}
    s = mod.summarise(per)
    assert s["zero_variance_share_diff_min"] == 0.25
    assert s["exploit_share_diff_max_abs"] == 0.02


@pytest.mark.skipif(not (ROOT / "data" / "chunks.jsonl").exists(), reason="no dataset")
def test_main_writes_results(tmp_path):
    out = tmp_path / "r.json"
    assert mod.main(["--group-size", "4", "--groups-per-task", "2", "--seeds", "7",
                     "--out", str(out)]) == 0
    assert "started_at" in out.read_text()
