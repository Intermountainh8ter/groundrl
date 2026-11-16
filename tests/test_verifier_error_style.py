from pathlib import Path

import pytest

from groundrl.dataset import load_dataset
from groundrl.verifier_error_style import run_verifier_error_style

DATA = Path(__file__).resolve().parents[1] / "data"


def test_bad_args():
    with pytest.raises(ValueError):
        run_verifier_error_style([], None, group_size=1)
    with pytest.raises(ValueError):
        run_verifier_error_style([], None, acc_styled=1.5)


@pytest.mark.skipif(not (DATA / "chunks.jsonl").exists(), reason="no dataset")
def test_deterministic_and_keys():
    ds = load_dataset(DATA)
    kw = dict(group_size=4, groups_per_task=3, seed=7)
    a = run_verifier_error_style(ds.tasks["test"][:20], ds.index, **kw)
    assert a == run_verifier_error_style(ds.tasks["test"][:20], ds.index, **kw)
    assert set(a["arms"]) == {"independent", "style_base", "style_raised"}
