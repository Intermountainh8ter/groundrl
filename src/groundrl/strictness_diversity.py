"""SIMULATED check: output diversity among full-reward rollouts per strictness rung.

No model is involved. A rollout is the scripted exploit policy with probability ``p`` and the
honest policy otherwise. For each rung, the distinct output strings among rollouts that score 1.0
are counted per group, averaged over groups with at least one full-reward rollout. The paired
bootstrap resamples tasks and recomputes the ratio of summed distinct counts to valid groups.
"""
from __future__ import annotations

import random
from collections.abc import Sequence

import numpy as np

from .ladder import VERSIONS, exploit_policy, grade, honest_policy
from .tasks import ChunkIndex, Task


def _mean_distinct(arr: np.ndarray) -> float:
    """``arr`` columns: sum of distinct counts, number of valid groups (per task)."""
    valid = arr[:, 1].sum()
    return float(arr[:, 0].sum() / valid) if valid > 0 else 0.0


def run_strictness_diversity(
    tasks: Sequence[Task],
    index: ChunkIndex,
    *,
    group_size: int = 8,
    groups_per_task: int = 20,
    probs: Sequence[float] = (0.1, 0.5, 0.9),
    seed: int = 7,
    n_boot: int = 1000,
    weak_rung: str = "first_number",
) -> dict:
    if group_size < 2 or groups_per_task < 1:
        raise ValueError("group_size must be >= 2 and groups_per_task >= 1")
    if weak_rung not in VERSIONS:
        raise ValueError(f"unknown rung {weak_rung!r}")
    rng = random.Random(seed)
    boot_rng = np.random.default_rng(seed)
    table = []
    for t in tasks:
        h, e = honest_policy(t, index), exploit_policy(t, index)
        table.append((h == e, {v: (grade(v, t, h), grade(v, t, e)) for v in VERSIONS}))
    n = len(table)
    by_level: dict[str, dict] = {}
    for p in probs:
        per_rung = {v: np.zeros((n, 2)) for v in VERSIONS}
        for i, (same, rewards) in enumerate(table):
            for _ in range(groups_per_task):
                n_ex = sum(rng.random() < p for _ in range(group_size))
                for v in VERSIONS:
                    rh, re_ = rewards[v]
                    ex_in = n_ex > 0 and re_ == 1.0
                    ho_in = n_ex < group_size and rh == 1.0
                    if not (ex_in or ho_in):
                        continue
                    distinct = 1 if (same and ex_in and ho_in) else int(ex_in) + int(ho_in)
                    per_rung[v][i] += (distinct, 1)
        cell: dict = {"mean_distinct": {v: _mean_distinct(per_rung[v]) for v in VERSIONS},
                      "valid_group_share": {
                          v: float(per_rung[v][:, 1].sum() / (n * groups_per_task)) if n else 0.0
                          for v in VERSIONS}}
        diff = _mean_distinct(per_rung[weak_rung]) - _mean_distinct(per_rung["strict"])
        lo = hi = diff
        if n:
            idx = boot_rng.integers(0, n, size=(n_boot, n))
            boots = np.array([_mean_distinct(per_rung[weak_rung][r])
                              - _mean_distinct(per_rung["strict"][r]) for r in idx])
            lo, hi = (float(x) for x in np.percentile(boots, [2.5, 97.5]))
        cell[f"{weak_rung}_minus_strict"] = diff
        cell[f"{weak_rung}_minus_strict_ci95"] = [lo, hi]
        by_level[f"p_{p:g}"] = cell
    return {
        "n_tasks": n,
        "group_size": group_size,
        "groups_per_task": groups_per_task,
        "versions": list(VERSIONS),
        "exploit_probs": list(probs),
        "weak_rung": weak_rung,
        "strictness": {"diversity_by_level": by_level},
    }
