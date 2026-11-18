import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import simulated_grpo_crossover_seeds as m  # noqa: E402


def test_spread_small_grid():
    r = m.spread([7, 11], honest=(0.3, 0.9), tasks=5, steps=50)
    assert set(r["per_seed"]) == {"7", "11"}
    assert r["max_minus_min"] is None or r["max_minus_min"] >= 0


def test_grid():
    assert m.HONEST[0] == 0.5 and m.HONEST[-1] == 0.7 and len(m.HONEST) == 9
