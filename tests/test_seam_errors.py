from pathlib import Path

import pytest

from groundrl.dataset import load_dataset
from groundrl.seam_errors import group_outcome, run_seam_errors

DATA = Path(__file__).resolve().parents[1] / "data"


def test_group_outcome_cases():
    ok = [True, True, False, False]
    assert group_outcome(1.0, 0.0, ok, [False] * 4) == (False, False)
    # partial false positives raise wrong rollouts' advantage but never above zero: no flip
    one = [True, False, False, False]
    assert group_outcome(1.0, 0.0, one, [False, True, False, False]) == (False, False)
    # every wrong rollout accepted: all-pass group is degenerate
    assert group_outcome(1.0, 0.0, one, [False, True, True, True]) == (False, True)
    # shared seam on every wrong rollout makes the group all-pass: degenerate, no flip
    assert group_outcome(1.0, 0.0, ok, [True] * 4) == (False, True)


@pytest.mark.skipif(not (DATA / "chunks.jsonl").exists(), reason="no dataset")
def test_deterministic_and_zero_fp():
    ds = load_dataset(DATA)
    kw = dict(group_size=8, groups_per_task=5, seed=7, n_boot=50)
    a = run_seam_errors(ds.tasks["test"][:20], ds.index, **kw)
    assert a == run_seam_errors(ds.tasks["test"][:20], ds.index, **kw)
    z = run_seam_errors(ds.tasks["test"][:20], ds.index, fp_rate=0.0, **kw)
    assert all(v == 0.0 for v in z["advantage_sign_flip_share"].values())


def test_bad_args():
    with pytest.raises(ValueError):
        run_seam_errors([], None, group_size=1)
    with pytest.raises(ValueError):
        run_seam_errors([], None, fp_rate=1.5)
