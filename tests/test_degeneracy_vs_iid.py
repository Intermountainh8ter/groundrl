from pathlib import Path

import pytest

from groundrl.dataset import load_dataset
from groundrl.degeneracy_vs_iid import pooled_share, run_degeneracy_vs_iid, task_share

DATA = Path(__file__).resolve().parents[1] / "data"


def test_formulas():
    assert pooled_share(0.5, 2) == 0.5
    assert task_share(1.0, 1.0, 0.3, 8) == 1.0
    assert task_share(0.0, 1.0, 0.5, 2) == 0.5


def test_bad_args():
    with pytest.raises(ValueError):
        run_degeneracy_vs_iid([], None, group_size=1)


@pytest.mark.skipif(not (DATA / "chunks.jsonl").exists(), reason="no dataset")
def test_deterministic_and_keys():
    ds = load_dataset(DATA)
    kw = dict(group_size=4, groups_per_task=3, probs=(0.5,), seed=7, n_boot=50)
    a = run_degeneracy_vs_iid(ds.tasks["test"][:20], ds.index, **kw)
    assert a == run_degeneracy_vs_iid(ds.tasks["test"][:20], ds.index, **kw)
    assert "pooled_minus_heterogeneous_max_gap" in a["degeneracy"]
