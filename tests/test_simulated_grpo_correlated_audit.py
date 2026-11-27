import importlib.util
import json
from pathlib import Path

_p = Path(__file__).resolve().parents[1] / "scripts" / "simulated_grpo_correlated_audit.py"
_spec = importlib.util.spec_from_file_location("sgca", _p)
sgca = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sgca)


def test_deterministic():
    assert sgca.run_arms(7, tasks=5, steps=20) == sgca.run_arms(7, tasks=5, steps=20)


def test_miss_kinds():
    assert sgca._miss("const", 0.25, 0.9) == 0.25
    assert sgca._miss("prop", 1.0, 0.4) == 0.4


def test_main_writes_provenance(tmp_path):
    out = tmp_path / "r.json"
    assert sgca.main(["--seed", "7", "--out", str(out)]) == 0
    d = json.loads(out.read_text())
    assert "started_at" in d["provenance"]
    assert "exploit_prob_final" in d["arms"]["miss_prop_p"]
