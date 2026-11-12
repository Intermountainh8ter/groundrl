import random
from pathlib import Path

import pytest

from groundrl.dataset import load_dataset
from groundrl.fuzz_candidates import MUTATIONS, mutate, run_fuzz, wilson_interval
from groundrl.ladder import honest_policy

DATA = Path(__file__).resolve().parents[1] / "data"


def test_wilson_bounds():
    lo, hi = wilson_interval(5, 10)
    assert 0 <= lo < 0.5 < hi <= 1
    assert wilson_interval(0, 0) == (0.0, 0.0)


@pytest.mark.skipif(not (DATA / "chunks.jsonl").exists(), reason="no dataset")
def test_run_fuzz_shape_and_determinism():
    ds = load_dataset(DATA)
    tasks = ds.tasks["test"][:40]
    a = run_fuzz(tasks, ds.index, 7)
    assert a == run_fuzz(tasks, ds.index, 7)
    r = a["exploit_candidate_rate"]["by_rung"]
    assert r["strict"] == 0.0
    assert a["n_mutated_outputs"] == 40 * len(MUTATIONS)
    t = next(x for x in tasks if x.answerable)
    with pytest.raises(ValueError):
        mutate("nope", t, honest_policy(t, ds.index), random.Random(0))
