import importlib.util
import json
from pathlib import Path

_p = Path(__file__).resolve().parents[1] / "scripts" / "simulated_grpo_isomorphic_arm.py"
_spec = importlib.util.spec_from_file_location("sgia", _p)
sgia = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sgia)


def test_deterministic():
    assert sgia.run_arms(7, tasks=4, steps=20) == sgia.run_arms(7, tasks=4, steps=20)


def test_isomorphic_accept_scaled():
    r = sgia.run_arms(7, tasks=2, steps=5, survival=0.5)
    assert abs(r["arms"]["isomorphic"]["exploit_accept"] - 0.3) < 1e-9
    assert r["arms"]["extensional"]["exploit_accept"] == 0.6


def test_main_writes_provenance(tmp_path):
    out = tmp_path / "r.json"
    assert sgia.main(["--seed", "1", "--out", str(out)]) == 0
    d = json.loads(out.read_text())
    assert "started_at" in d["provenance"]
    assert 0.0 <= d["arms"]["isomorphic"]["final_exploit_prob"] <= 1.0
