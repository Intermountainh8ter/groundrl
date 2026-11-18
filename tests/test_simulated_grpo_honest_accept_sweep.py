import importlib.util
from pathlib import Path

_p = Path(__file__).resolve().parents[1] / "scripts" / "simulated_grpo_honest_accept_sweep.py"
_spec = importlib.util.spec_from_file_location("honest_sweep", _p)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def test_small_sweep_structure_and_determinism():
    kw = dict(honest=(0.3, 0.7), tasks=5, steps=20)
    a = mod.run_sweep(7, **kw)
    assert a == mod.run_sweep(7, **kw)
    assert set(a["sweep"]) == {"exploit_prob_final_h0p30", "exploit_prob_final_h0p70"}
    assert all(0.0 <= v <= 1.0 for v in a["sweep"].values())
    assert "min_honest_accept_exploit_lt_0p5" in a["crossover"]


def test_main_writes_provenance(tmp_path):
    out = tmp_path / "r.json"
    assert mod.main(["--seed", "1", "--out", str(out)]) == 0
    assert '"started_at"' in out.read_text()
