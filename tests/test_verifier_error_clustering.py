import random
from pathlib import Path

import pytest

from groundrl.dataset import load_dataset
from groundrl.verifier_error_clustering import blind_spots, run_verifier_error_clustering

DATA = Path(__file__).resolve().parents[1] / "data"


def test_blind_spots_exact_count():
    assert sum(blind_spots(random.Random(1), 50, 0.2)) == 10
    assert sum(blind_spots(random.Random(1), 50, 0.0)) == 0


def test_bad_args():
    with pytest.raises(ValueError):
        run_verifier_error_clustering([], None, group_size=1)
    with pytest.raises(ValueError):
        run_verifier_error_clustering([], None, error_rate=-0.1)


@pytest.mark.skipif(not (DATA / "chunks.jsonl").exists(), reason="no dataset")
def test_deterministic_and_keys():
    ds = load_dataset(DATA)
    kw = dict(group_size=4, groups_per_task=3, seed=7)
    a = run_verifier_error_clustering(ds.tasks["test"][:20], ds.index, **kw)
    assert a == run_verifier_error_clustering(ds.tasks["test"][:20], ds.index, **kw)
    assert a["n_blind_spots"] == 4
    assert 0.0 <= a["modes"]["clustered"]["zero_variance_group_share"] <= 1.0
