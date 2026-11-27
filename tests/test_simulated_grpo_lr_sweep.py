import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import simulated_grpo_lr_sweep as m  # noqa: E402


def test_arm_shape_and_determinism():
    a = m.run_arm(7, tasks=5, steps=20)
    b = m.run_arm(7, tasks=5, steps=20)
    assert a == b
    assert set(a["final_exploit_prob"]) == {m._key(x) for x in m.LRS}
    assert isinstance(a["hypothesis_supported"], bool)


def test_all_accepting_grader_does_not_move():
    rng = random.Random(1)
    kw = dict(group=8, steps=5, lr=0.5, p0=0.05, honest_accept=1.0, exploit_accept=1.0)
    assert abs(m.run_task(rng, **kw) - 0.05) < 1e-9


def test_main_writes_provenance(tmp_path):
    out = tmp_path / "r.json"
    assert m.main(["--seed", "7", "--out", str(out)]) == 0
    assert '"started_at"' in out.read_text()
