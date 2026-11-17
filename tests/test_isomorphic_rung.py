from dataclasses import replace

import pytest

from groundrl.isomorphic_rung import memorizer_policy, perturb, run_isomorphic_rung


def test_empty():
    with pytest.raises(ValueError):
        run_isomorphic_rung([], None)


def test_perturb_reverses_and_is_involution(dataset):
    t = next(x for x in dataset.tasks["test"] if x.answerable and len(set(x.gold_tokens)) > 1)
    assert perturb(t).gold_tokens == list(reversed(t.gold_tokens))
    assert perturb(perturb(t)).gold_tokens == t.gold_tokens
    assert perturb(replace(t, gold_tokens=["a"])).gold_tokens == ["a"]


def test_memorizer_fails_iso_on_distinct_multi(dataset):
    res = run_isomorphic_rung(dataset.tasks["test"], dataset.index)
    assert res["memorizer_accept_rate"]["strict"] == 1.0
    assert res["honest_accept_rate"]["strict_iso"] == 1.0
    assert res["memorizer_accept_rate"]["strict_iso"] < 1.0
    t = next(x for x in dataset.tasks["test"] if x.answerable and len(set(x.gold_tokens)) > 1)
    assert memorizer_policy(t, dataset.index)
