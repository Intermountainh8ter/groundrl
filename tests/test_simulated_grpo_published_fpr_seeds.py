import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import simulated_grpo_published_fpr_seeds as m  # noqa: E402


def test_spread_small_grid():
    r = m.spread([7, 11], 0.832, honest=(0.3, 0.9), tasks=5, steps=50)
    assert set(r["per_seed"]) == {"7", "11"}
    assert r["max_minus_min"] is None or r["max_minus_min"] >= 0


def test_grid():
    assert m.HONEST[0] == 0.6 and m.HONEST[-1] == 0.9 and len(m.HONEST) == 31
    assert m.EXPLOITS == (0.832, 0.869)
