from pathlib import Path

import pytest

from groundrl.dataset import load_dataset
from groundrl.group_degeneracy import p_key, run_group_degeneracy

DATA = Path(__file__).resolve().parents[1] / "data"
pytestmark = pytest.mark.skipif(not (DATA / "chunks.jsonl").exists(), reason="no dataset")


@pytest.fixture(scope="module")
def ds():
    return load_dataset(DATA)


def test_p_zero_and_one_are_fully_degenerate(ds):
    r = run_group_degeneracy(ds.tasks["test"][:20], ds.index, probs=(0.0, 1.0), groups_per_task=3)
    for key in (p_key(0.0), p_key(1.0)):
        assert all(s == 1.0 for s in r["zero_variance_group_share"][key].values())


def test_deterministic_and_strict_rarely_degenerate(ds):
    kw = dict(group_size=8, groups_per_task=5, probs=(0.5,), seed=7)
    a = run_group_degeneracy(ds.tasks["test"][:30], ds.index, **kw)
    b = run_group_degeneracy(ds.tasks["test"][:30], ds.index, **kw)
    assert a == b
    assert a["zero_variance_group_share"]["p_0.5"]["strict"] < 0.2


def test_bad_args(ds):
    with pytest.raises(ValueError):
        run_group_degeneracy(ds.tasks["test"][:2], ds.index, group_size=1)
