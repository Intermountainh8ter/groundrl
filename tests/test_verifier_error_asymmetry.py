import random
from pathlib import Path

import pytest

from groundrl.dataset import load_dataset
from groundrl.verifier_error_asymmetry import _rollout, run_verifier_error_asymmetry

DATA = Path(__file__).resolve().parents[1] / "data"


def test_rollout_directions():
    rng = random.Random(1)
    # FP never touches honest rollouts; FN never touches exploit rollouts.
    assert _rollout(rng, 1.0, 0.0, 0.0, 1.0, "fp") == (False, True)
    assert _rollout(rng, 1.0, 0.0, 1.0, 1.0, "fp") == (True, True)
    assert _rollout(rng, 1.0, 0.0, 0.0, 1.0, "fn") == (False, False)
    assert _rollout(rng, 1.0, 0.0, 1.0, 1.0, "fn") == (True, False)
    assert _rollout(rng, 1.0, 0.0, 1.0, 0.0, "fp") == (True, False)


def test_bad_args():
    with pytest.raises(ValueError):
        run_verifier_error_asymmetry([], None, group_size=1)
    with pytest.raises(ValueError):
        run_verifier_error_asymmetry([], None, error_rate=1.5)


@pytest.mark.skipif(not (DATA / "chunks.jsonl").exists(), reason="no dataset")
def test_deterministic_and_keys():
    ds = load_dataset(DATA)
    kw = dict(group_size=4, groups_per_task=3, probs=(0.5,), seed=7, n_boot=50)
    a = run_verifier_error_asymmetry(ds.tasks["test"][:20], ds.index, **kw)
    assert a == run_verifier_error_asymmetry(ds.tasks["test"][:20], ds.index, **kw)
    assert "exploit_share_fp_minus_fn" in a["asymmetry"]
    c = a["cells"]["p_0.5"]["strict"]
    assert 0.0 <= c["fp"]["exploit_share"] <= 1.0
    assert c["exploit_share_fp_minus_fn_ci95"][0] <= c["exploit_share_fp_minus_fn_ci95"][1]
