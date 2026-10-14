from __future__ import annotations

import math

import numpy as np

from groundrl.stats import bootstrap_ci, mean_ci


def test_bootstrap_interval_brackets_the_point_estimate():
    values = [0.0, 1.0] * 50
    point, lo, hi = bootstrap_ci(values, n_boot=500, seed=1)
    assert point == 0.5 and lo < 0.5 < hi and 0.35 < lo and hi < 0.65


def test_bootstrap_is_deterministic_for_a_seed_and_varies_across_seeds():
    values = list(np.linspace(0, 1, 40))
    assert bootstrap_ci(values, seed=3) == bootstrap_ci(values, seed=3)
    assert bootstrap_ci(values, seed=3) != bootstrap_ci(values, seed=4)


def test_degenerate_inputs():
    assert all(math.isnan(x) for x in bootstrap_ci([]))
    assert bootstrap_ci([0.7]) == (0.7, 0.7, 0.7)
    point, lo, hi = bootstrap_ci([1.0] * 10)
    assert point == lo == hi == 1.0


def test_custom_statistic():
    point, lo, hi = bootstrap_ci([1, 2, 3, 4, 100], stat=np.median, n_boot=200, seed=0)
    assert point == 3 and lo <= point <= hi


def test_mean_ci_dict():
    d = mean_ci([0.0, 1.0, 1.0, 1.0], n_boot=100)
    assert set(d) == {"mean", "lo", "hi", "n"} and d["mean"] == 0.75 and d["n"] == 4
