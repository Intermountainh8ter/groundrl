from pathlib import Path

import pytest

from groundrl.dataset import load_dataset
from groundrl.leakiness_vs_hack_rate import run_leakiness_vs_hack_rate

DATA = Path(__file__).resolve().parents[1] / "data"


def test_bad_args():
    with pytest.raises(ValueError):
        run_leakiness_vs_hack_rate([], None, group_size=1)
    with pytest.raises(ValueError):
        run_leakiness_vs_hack_rate([], None, prob=1.5)


@pytest.mark.skipif(not (DATA / "chunks.jsonl").exists(), reason="no dataset")
def test_small_run_shape_and_strict_zero():
    ds = load_dataset(DATA)
    res = run_leakiness_vs_hack_rate(ds.tasks["test"][:20], ds.index, groups_per_task=2)
    assert res["leakiness"]["n_cells"] == 20
    assert -1.0 <= res["leakiness"]["spearman_vs_hack_rate"] <= 1.0
    for c, v in res["exploit_share_among_rewarded"].items():
        if c.startswith("strict/"):
            assert v == 0.0
