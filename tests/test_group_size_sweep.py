from pathlib import Path

import pytest

from groundrl.dataset import load_dataset
from groundrl.exploit_advantage import run_group_size_sweep

DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.mark.skipif(not (DATA / "chunks.jsonl").exists(), reason="no dataset")
def test_sweep_deterministic_and_keyed():
    ds = load_dataset(DATA)
    kw = dict(group_sizes=(2, 4), groups_per_task=3, seed=7)
    a = run_group_size_sweep(ds.tasks["test"][:20], ds.index, **kw)
    assert a == run_group_size_sweep(ds.tasks["test"][:20], ds.index, **kw)
    share = a["exploit_advantage_positive_share"]
    assert set(share) == {"G_2", "G_4"}
    assert all(0.0 <= s <= 1.0 for s in share["G_4"]["q_0.5"]["p_0.5"].values())


def test_sweep_rejects_empty_sizes():
    with pytest.raises(ValueError):
        run_group_size_sweep([], None, group_sizes=())
