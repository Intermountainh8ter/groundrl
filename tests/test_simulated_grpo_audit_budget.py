import importlib.util
import json
from pathlib import Path

_p = Path(__file__).resolve().parents[1] / "scripts" / "simulated_grpo_audit_budget.py"
_spec = importlib.util.spec_from_file_location("sgab", _p)
sgab = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sgab)


def test_deterministic():
    a = sgab.run_sweep(7, tasks=5, steps=20)
    b = sgab.run_sweep(7, tasks=5, steps=20)
    assert a == b


def test_full_audit_suppresses_exploit():
    r = sgab.run_sweep(7, budgets=(1.0,), tasks=20, steps=100)
    assert r["sweep"]["exploit_prob_final_b1"] < 0.05


def test_main_writes_provenance(tmp_path):
    out = tmp_path / "r.json"
    assert sgab.main(["--seed", "7", "--out", str(out)]) == 0
    d = json.loads(out.read_text())
    assert "started_at" in d["provenance"]
    assert "exploit_prob_final_b0p25" in d["sweep"]
