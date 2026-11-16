from pathlib import Path

import pytest

from groundrl.dataset import load_dataset
from groundrl.strictness_diversity import run_strictness_diversity

DATA = Path(__file__).resolve().parents[1] / "data"
pytestmark = pytest.mark.skipif(not (DATA / "chunks.jsonl").exists(), reason="no dataset")


@pytest.fixture(scope="module")
def ds():
    return load_dataset(DATA)


def test_pure_policies_have_one_distinct_output(ds):
    r = run_strictness_diversity(ds.tasks["test"][:20], ds.index, probs=(0.0, 1.0),
                                 groups_per_task=3, n_boot=10)
    for cell in r["strictness"]["diversity_by_level"].values():
        assert all(v <= 1.0 for v in cell["mean_distinct"].values())


def test_deterministic_and_bounded(ds):
    kw = dict(group_size=8, groups_per_task=5, probs=(0.5,), seed=7, n_boot=20)
    a = run_strictness_diversity(ds.tasks["test"][:30], ds.index, **kw)
    b = run_strictness_diversity(ds.tasks["test"][:30], ds.index, **kw)
    assert a == b
    cell = a["strictness"]["diversity_by_level"]["p_0.5"]
    assert all(0.0 <= v <= 2.0 for v in cell["mean_distinct"].values())
    assert len(cell["first_number_minus_strict_ci95"]) == 2


def test_bad_args(ds):
    with pytest.raises(ValueError):
        run_strictness_diversity(ds.tasks["test"][:2], ds.index, group_size=1)
    with pytest.raises(ValueError):
        run_strictness_diversity(ds.tasks["test"][:2], ds.index, weak_rung="nope")
