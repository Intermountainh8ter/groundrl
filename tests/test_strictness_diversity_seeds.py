import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "strictness_diversity_seeds.py"


def _load():
    spec = importlib.util.spec_from_file_location("sds", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fake(gap):
    return {"strictness": {"diversity_by_level": {"p_0.5": {"first_number_minus_strict": gap}}}}


def test_summarise_mean_and_spread():
    out = _load().summarise({"7": _fake(0.9), "13": _fake(1.0)}, [0.5])
    assert abs(out["gap_p0.5_mean"] - 0.95) < 1e-12
    assert abs(out["gap_p0.5_spread"] - 0.1) < 1e-12
    assert out["per_seed"]["7"]["gap_p0.5"] == 0.9
