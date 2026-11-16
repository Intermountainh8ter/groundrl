"""SIMULATED check: zero-variance group share under a pooled i.i.d. Bernoulli model versus
per-task success rates, against the sampled share from ``group_degeneracy``.

No model is involved. A rollout is the exploit policy with probability ``p`` and the honest
policy otherwise; its success is reward == 1.0. The pooled prediction uses one success rate
for all rollouts; the per-task prediction uses the exact probability that a group of a task
has identical rewards.
"""
from __future__ import annotations

import random
from collections.abc import Sequence

from .group_degeneracy import p_key
from .ladder import VERSIONS, exploit_policy, grade, honest_policy
from .tasks import ChunkIndex, Task

EPS = 1e-9


def pooled_share(q: float, g: int) -> float:
    return q**g + (1.0 - q) ** g


def task_share(rh: float, re_: float, p: float, g: int) -> float:
    """Exact probability that all g rewards in a group of one task are equal."""
    if rh == re_:
        return 1.0
    return p**g + (1.0 - p) ** g


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _boot(vals: Sequence[float], rng: random.Random, n_boot: int) -> list[float]:
    n = len(vals)
    if n == 0:
        return [0.0, 0.0]
    means = sorted(_mean([vals[rng.randrange(n)] for _ in range(n)]) for _ in range(n_boot))
    return [means[int(0.025 * (n_boot - 1))], means[int(0.975 * (n_boot - 1))]]


def run_degeneracy_vs_iid(
    tasks: Sequence[Task],
    index: ChunkIndex,
    *,
    group_size: int = 8,
    groups_per_task: int = 20,
    probs: Sequence[float] = (0.1, 0.5, 0.9),
    seed: int = 7,
    n_boot: int = 1000,
) -> dict:
    if group_size < 2 or groups_per_task < 1:
        raise ValueError("group_size must be >= 2 and groups_per_task >= 1")
    rng = random.Random(seed)
    table = []
    for t in tasks:
        h, e = honest_policy(t, index), exploit_policy(t, index)
        table.append({v: (grade(v, t, h), grade(v, t, e)) for v in VERSIONS})
    cells: dict[str, dict[str, dict]] = {}
    max_gap = float("-inf")
    inside = 0
    n_cells = 0
    for p in probs:
        cells[p_key(p)] = {}
        for v in VERSIONS:
            per_task, sim, qs = [], [], []
            for rewards in table:
                rh, re_ = rewards[v]
                per_task.append(task_share(rh, re_, p, group_size))
                qs.append(p * (re_ == 1.0) + (1 - p) * (rh == 1.0))
                zero = 0
                for _ in range(groups_per_task):
                    picks = [rng.random() < p for _ in range(group_size)]
                    zero += len({re_ if x else rh for x in picks}) == 1
                sim.append(zero / groups_per_task)
            pooled = pooled_share(_mean(qs), group_size)
            het = _mean(per_task)
            lo, hi = _boot(per_task, rng, n_boot)
            gap = pooled - het
            max_gap = max(max_gap, gap)
            ok = lo <= _mean(sim) <= hi
            inside += ok
            n_cells += 1
            cells[p_key(p)][v] = {
                "pooled_prediction": pooled,
                "per_task_prediction": het,
                "per_task_ci95": [lo, hi],
                "simulated": _mean(sim),
                "pooled_minus_per_task": gap,
                "simulated_inside_per_task_ci": ok,
            }
    return {
        "n_tasks": len(tasks),
        "group_size": group_size,
        "groups_per_task": groups_per_task,
        "versions": list(VERSIONS),
        "exploit_probs": list(probs),
        "cells": cells,
        "degeneracy": {
            "pooled_minus_heterogeneous_max_gap": max_gap,
            "cells_simulated_inside_ci": inside,
            "n_cells": n_cells,
            "pooled_exceeds_per_task_cells": sum(
                c["pooled_minus_per_task"] > EPS for pc in cells.values() for c in pc.values()),
        },
    }
