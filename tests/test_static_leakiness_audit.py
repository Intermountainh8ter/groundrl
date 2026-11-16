from pathlib import Path

import pytest

from groundrl.dataset import load_dataset
from groundrl.static_leakiness_audit import PROBES, run_static_audit, spearman

DATA = Path(__file__).resolve().parents[1] / "data"


def test_spearman_ties_and_sign():
    assert spearman([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([1, 1, 1], [1, 2, 3]) == 0.0


@pytest.mark.skipif(not (DATA / "chunks.jsonl").exists(), reason="no dataset")
def test_static_audit_shape_and_determinism():
    ds = load_dataset(DATA)
    tasks = ds.tasks["test"][:30]
    a = run_static_audit(tasks, ds.index, 7, n_boot=50)
    assert a == run_static_audit(tasks, ds.index, 7, n_boot=50)
    assert len(a["cells"]) == 20
    assert all(len(v) == 4 for v in PROBES.values())
    assert all(a["audit_acceptance_share"][f"strict/{m}"] == 0.0 for m in PROBES)
    lo, hi = a["rank_agreement"]["bootstrap_95"]
    assert lo <= hi
