from pathlib import Path

import numpy as np
import pytest

from groundrl.audit_allocation import _remaining_share, run_audit_allocation
from groundrl.dataset import load_dataset

DATA = Path(__file__).resolve().parents[1] / "data"


def _counts():
    # columns: exploit rewarded, honest rewarded, exploit ok, honest ok, eligible
    return np.array([[10.0, 10.0, 0.0, 1.0, 1.0], [0.0, 20.0, 1.0, 1.0, 0.0]])


def test_none_matches_raw_share():
    assert _remaining_share(_counts(), 0.5, "none") == pytest.approx(0.25)


def test_targeted_beats_random_when_exploits_are_eligible():
    c = _counts()
    assert _remaining_share(c, 0.1, "targeted") < _remaining_share(c, 0.1, "random")


def test_full_budget_equal():
    c = _counts()
    assert _remaining_share(c, 1.0, "random") == _remaining_share(c, 1.0, "targeted") == 0.0


def test_unknown_strategy_and_args():
    with pytest.raises(ValueError):
        _remaining_share(_counts(), 0.1, "bogus")
    with pytest.raises(ValueError):
        run_audit_allocation([], None, group_size=1)
    with pytest.raises(ValueError):
        run_audit_allocation([], None, audit_fraction=2.0)


@pytest.mark.skipif(not (DATA / "chunks.jsonl").exists(), reason="no dataset")
def test_deterministic_and_keys():
    ds = load_dataset(DATA)
    kw = dict(group_size=4, groups_per_task=3, probs=(0.5,), seed=7, n_boot=50)
    a = run_audit_allocation(ds.tasks["test"][:20], ds.index, **kw)
    assert a == run_audit_allocation(ds.tasks["test"][:20], ds.index, **kw)
    assert "exploit_share_random_minus_targeted" in a["audit"]
