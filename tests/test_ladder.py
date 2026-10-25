from pathlib import Path

import pytest

from groundrl.dataset import load_dataset
from groundrl.ladder import VERSIONS, exploit_policy, grade, honest_policy, run_ladder

DATA = Path(__file__).resolve().parents[1] / "data"
pytestmark = pytest.mark.skipif(not (DATA / "chunks.jsonl").exists(), reason="no dataset")


@pytest.fixture(scope="module")
def ds():
    return load_dataset(DATA)


def test_honest_passes_every_rung(ds):
    for t in ds.tasks["test"][:60]:
        for v in VERSIONS:
            assert grade(v, t, honest_policy(t, ds.index)) == 1.0


def test_strict_rejects_exploit(ds):
    for t in ds.tasks["test"][:60]:
        assert grade("strict", t, exploit_policy(t, ds.index)) < 1.0


def test_unknown_version(ds):
    with pytest.raises(ValueError):
        grade("nope", ds.tasks["test"][0], "x")


def test_run_ladder_shape_and_ordering(ds):
    res = run_ladder(ds.tasks["test"][:80], ds.index)
    s = res["exploit_success_rate"]["by_version"]
    assert s["strict"] == 0.0
    assert s["first_number"] >= s["strict"]
    z = res["zero_spread_share"]["by_version"]
    assert z["first_number"] >= z["strict"]
