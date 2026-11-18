import importlib.util
from pathlib import Path

_p = Path(__file__).resolve().parents[1] / "scripts" / "simulated_grpo_isomorphic_noisy.py"
_spec = importlib.util.spec_from_file_location("sgin", _p)
sgin = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sgin)


def test_deterministic():
    a = sgin.run_arms(7, tasks=3, steps=10, noises=(0.2,))
    assert a == sgin.run_arms(7, tasks=3, steps=10, noises=(0.2,))


def test_noisy_accept_formula():
    assert abs(sgin.noisy(0.6, 0.2) - 0.56) < 1e-9
    assert abs(sgin.noisy(0.18, 0.5) - 0.5) < 1e-9
    assert abs(sgin.noisy(0.3, 0.0) - 0.3) < 1e-9


def test_keys():
    r = sgin.run_arms(7, tasks=2, steps=5, noises=(0.2, 0.5))
    assert "noise_0p2" in r["arms"]["isomorphic_noisy"]
    assert "noise_0p5" in r["arms"]["extensional_noisy"]
