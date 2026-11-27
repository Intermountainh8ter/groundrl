import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import simulated_grpo_kl_term as m  # noqa: E402


def test_arm_shape_and_determinism():
    a = m.run_arm(7, tasks=5, steps=20)
    b = m.run_arm(7, tasks=5, steps=20)
    assert a == b
    assert set(a["final_exploit_prob"]) == {"b0p00", "b0p02", "b0p10"}
    assert isinstance(a["drop_at_least_min"], bool)


def test_pull_applies_with_zero_advantage():
    # all-accepting grader: every group has zero std, so only the pull moves theta
    import random
    rng = random.Random(1)
    kw = dict(group=8, steps=5, lr=0.5, p0=0.05, honest_accept=1.0, exploit_accept=1.0)
    assert abs(m.run_task(rng, 0.1, **kw) - 0.05) < 1e-9


def test_main_writes_provenance(tmp_path):
    out = tmp_path / "r.json"
    assert m.main(["--seed", "7", "--out", str(out)]) == 0
    assert '"started_at"' in out.read_text()
