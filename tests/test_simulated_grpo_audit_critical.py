import importlib.util
import json
from pathlib import Path

_p = Path(__file__).resolve().parents[1] / "scripts" / "simulated_grpo_audit_critical.py"
_spec = importlib.util.spec_from_file_location("sgac", _p)
sgac = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sgac)


def test_deterministic():
    a = sgac.run_critical(7, tasks=5, steps=20)
    b = sgac.run_critical(7, tasks=5, steps=20)
    assert a == b


def test_break_even():
    r = sgac.run_critical(7, budgets=(0.7,), tasks=3, steps=10)
    assert abs(r["critical"]["break_even_budget"] - 0.5) < 1e-9


def test_main_writes_provenance(tmp_path):
    out = tmp_path / "r.json"
    assert sgac.main(["--seed", "7", "--out", str(out)]) == 0
    d = json.loads(out.read_text())
    assert "started_at" in d["provenance"]
    assert "min_budget_below_0p25" in d["critical"]
