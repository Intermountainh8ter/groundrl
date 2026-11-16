from pathlib import Path

import pytest

from groundrl.dataset import load_dataset
from groundrl.degenerate_group_by_reward import run_degenerate_group_by_reward
from groundrl.example import make_examples

DATA = Path(__file__).resolve().parents[1] / "data"
pytestmark = pytest.mark.skipif(not (DATA / "chunks.jsonl").exists(), reason="no dataset")


def _examples(n):
    ds = load_dataset(DATA)
    return make_examples(ds.tasks["test"][:n], ds.index)


def test_deterministic_and_bounded():
    kw = dict(group_size=8, groups_per_task=3, seed=7)
    a = run_degenerate_group_by_reward(_examples(15), **kw)
    assert a == run_degenerate_group_by_reward(_examples(15), **kw)
    for v in a["versions"]:
        assert 0.0 <= a["degenerate_group_fraction_by_reward"][v]["mean_share"] <= 1.0


def test_prob_extremes_fully_degenerate():
    for p in (0.0, 1.0):
        r = run_degenerate_group_by_reward(_examples(8), hacks=("copy_chunk",), prob=p,
                                           groups_per_task=2)
        assert all(r["degenerate_group_fraction_by_reward"][v]["mean_share"] == 1.0
                   for v in r["versions"])


def test_bad_args():
    with pytest.raises(ValueError):
        run_degenerate_group_by_reward(_examples(2), group_size=1)
    with pytest.raises(ValueError):
        run_degenerate_group_by_reward(_examples(2), hacks=())
