import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import simulated_grpo_published_fpr_sweep as m  # noqa: E402


def test_keys_and_determinism():
    kw = dict(honest=(0.5, 1.0), tasks=5, steps=20)
    a = m.run(7, **kw)
    b = m.run(7, **kw)
    assert a == b
    assert set(a["crossover_by_exploit_accept"]) == {"e0p83", "e0p87"}


def test_grid_and_exploits():
    assert m.EXPLOIT == (0.832, 0.869)
    assert m.HONEST[0] == 0.5 and m.HONEST[-1] == 1.0
