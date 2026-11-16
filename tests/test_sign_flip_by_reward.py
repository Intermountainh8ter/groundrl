from pathlib import Path

import pytest

from groundrl.dataset import load_dataset
from groundrl.example import make_examples
from groundrl.sign_flip_by_reward import run_sign_flip_by_reward

DATA = Path(__file__).resolve().parents[1] / "data"
pytestmark = pytest.mark.skipif(not (DATA / "chunks.jsonl").exists(), reason="no dataset")


def _examples(n):
    ds = load_dataset(DATA)
    return make_examples(ds.tasks["test"][:n], ds.index)


def test_deterministic_and_bounded():
    kw = dict(group_size=8, groups_per_task=3, seed=7)
    a = run_sign_flip_by_reward(_examples(15), **kw)
    assert a == run_sign_flip_by_reward(_examples(15), **kw)
    for r in a["sign_flip_share"].values():
        assert 0.0 <= r["mean_strict_reversal_share"] <= r["mean_share"] <= 1.0


def test_prob_one_has_no_flip():
    r = run_sign_flip_by_reward(_examples(8), hacks=("copy_chunk",), prob=1.0, groups_per_task=2)
    assert r["sign_flip_share"]["v0_vs_v2"]["mean_share"] == 0.0


def test_bad_args():
    with pytest.raises(ValueError):
        run_sign_flip_by_reward(_examples(2), group_size=1)
    with pytest.raises(ValueError):
        run_sign_flip_by_reward(_examples(2), hacks=())
