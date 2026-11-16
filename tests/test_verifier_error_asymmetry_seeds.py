import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verifier_error_asymmetry_seeds.py"


def _load():
    spec = importlib.util.spec_from_file_location("seeds_asym", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fake(x):
    return {"asymmetry": {"exploit_share_fp_minus_fn": x}}


def test_summarise_min_mean_spread():
    out = _load().summarise({"7": _fake(0.2), "13": _fake(0.1), "21": _fake(0.3)})
    assert out["primary_min"] == 0.1
    assert out["primary_max"] == 0.3
    assert abs(out["primary_mean"] - 0.2) < 1e-12
    assert abs(out["primary_spread"] - 0.2) < 1e-12
    assert out["per_seed"]["13"] == 0.1
