import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import simulated_grpo_strict_arm as m  # noqa: E402


def test_arm_shape_and_determinism():
    a = m.run_arm(7, tasks=5, steps=20)
    b = m.run_arm(7, tasks=5, steps=20)
    assert a == b
    assert set(a["exploit_prob_final"]) == {"e0p00", "e0p10", "e0p20"}
    assert "h0p90" in a["exploit_prob_final"]["e0p00"]
    assert isinstance(a["strict_stays_below_p0"], bool)


def test_main_writes_provenance(tmp_path):
    out = tmp_path / "r.json"
    assert m.main(["--seed", "7", "--out", str(out)]) == 0
    assert '"started_at"' in out.read_text()
