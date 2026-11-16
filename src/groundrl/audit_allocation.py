"""SIMULATED check: targeted versus random strong-verifier audit at equal budget.

No model is involved. A rollout is the exploit policy with probability ``p`` and the honest
policy otherwise; the weak rung (default ``first_number``) grades it. Rollouts with reward 1.0
are rewarded. A budget of ``audit_fraction`` of all rewarded rollouts is spent re-grading
rewarded rollouts with the strict rung; an audited rollout the strict rung does not score 1.0
loses its reward. Random audit spreads the budget uniformly over rewarded rollouts. Targeted
audit spends it first on rewarded rollouts of tasks where the weak and strict rung can differ
on the exploit policy (a property of the task and the rungs, not of any rollout label) and
spreads any surplus over the rest. Expected catches are computed in closed form from the
sampled counts, so the only randomness is the rollout sampling and the task bootstrap.
"""
from __future__ import annotations

import random
from collections.abc import Sequence

import numpy as np

from .ladder import VERSIONS, exploit_policy, grade, honest_policy
from .tasks import ChunkIndex, Task


def _remaining_share(counts: np.ndarray, frac: float, strategy: str) -> float:
    """Exploit share of rewarded rollouts left after an audit.

    ``counts`` columns: exploit rewarded, honest rewarded, exploit survives strict,
    honest survives strict, eligible flag.
    """
    ex, ho, ex_ok, ho_ok, elig = (counts[:, i] for i in range(5))
    rewarded = ex + ho
    budget = frac * rewarded.sum()
    if rewarded.sum() <= 0:
        return 0.0
    if strategy == "none":
        cover = np.zeros(len(counts))
    elif strategy == "random":
        cover = np.full(len(counts), min(1.0, frac))
    elif strategy == "targeted":
        pool = float((rewarded * elig).sum())
        c1 = min(1.0, budget / pool) if pool > 0 else 0.0
        spare = max(budget - pool, 0.0)
        rest = float((rewarded * (1 - elig)).sum())
        c0 = min(1.0, spare / rest) if rest > 0 else 0.0
        cover = np.where(elig > 0, c1, c0)
    else:
        raise ValueError(f"unknown strategy {strategy!r}")
    ex_left = (ex * (1 - cover * (1 - ex_ok))).sum()
    ho_left = (ho * (1 - cover * (1 - ho_ok))).sum()
    total = ex_left + ho_left
    return float(ex_left / total) if total > 0 else 0.0


def run_audit_allocation(
    tasks: Sequence[Task],
    index: ChunkIndex,
    *,
    group_size: int = 8,
    groups_per_task: int = 20,
    audit_fraction: float = 0.1,
    probs: Sequence[float] = (0.1, 0.5, 0.9),
    rungs: Sequence[str] = ("first_number", "dup_key", "skip_denominator"),
    seed: int = 7,
    n_boot: int = 1000,
    primary_rung: str = "first_number",
    primary_prob: float = 0.5,
) -> dict:
    if group_size < 2 or groups_per_task < 1:
        raise ValueError("group_size must be >= 2 and groups_per_task >= 1")
    if not 0.0 <= audit_fraction <= 1.0:
        raise ValueError("audit_fraction must be in [0, 1]")
    if primary_rung not in rungs or primary_prob not in probs:
        raise ValueError("primary cell must be one of the swept cells")
    rng = random.Random(seed)
    boot_rng = np.random.default_rng(seed)
    table = []
    for t in tasks:
        h, e = honest_policy(t, index), exploit_policy(t, index)
        table.append({v: (grade(v, t, h), grade(v, t, e)) for v in VERSIONS})
    n = len(table)
    n_roll = group_size * groups_per_task
    cells: dict[str, dict[str, dict]] = {}
    for p in probs:
        cells[f"p_{p}"] = {}
        for rung in rungs:
            rows = []
            for rw in table:
                rh, re_ = rw[rung]
                sh, se = rw["strict"]
                n_ex = sum(rng.random() < p for _ in range(n_roll))
                ex_r = n_ex if re_ == 1.0 else 0
                ho_r = (n_roll - n_ex) if rh == 1.0 else 0
                elig = 1.0 if (re_ == 1.0 and se != 1.0) else 0.0
                rows.append((ex_r, ho_r, float(se == 1.0), float(sh == 1.0), elig))
            arr = np.array(rows, dtype=float).reshape(-1, 5)
            shares = {s: _remaining_share(arr, audit_fraction, s)
                      for s in ("none", "random", "targeted")}
            diff = shares["random"] - shares["targeted"]
            lo = hi = diff
            if n:
                idx = boot_rng.integers(0, n, size=(n_boot, n))
                boots = np.array([_remaining_share(arr[r], audit_fraction, "random")
                                  - _remaining_share(arr[r], audit_fraction, "targeted")
                                  for r in idx])
                lo, hi = (float(x) for x in np.percentile(boots, [2.5, 97.5]))
            cells[f"p_{p}"][rung] = {
                "exploit_share": shares,
                "exploit_share_random_minus_targeted": diff,
                "exploit_share_random_minus_targeted_ci95": [lo, hi],
                "eligible_task_share": float(arr[:, 4].mean()) if n else 0.0,
            }
    primary = cells[f"p_{primary_prob}"][primary_rung]
    return {
        "n_tasks": n,
        "group_size": group_size,
        "groups_per_task": groups_per_task,
        "audit_fraction": audit_fraction,
        "rungs": list(rungs),
        "exploit_probs": list(probs),
        "cells": cells,
        "audit": {
            "primary_cell": {"rung": primary_rung, "exploit_prob": primary_prob,
                             "audit_fraction": audit_fraction},
            "exploit_share_random_minus_targeted":
                primary["exploit_share_random_minus_targeted"],
            "exploit_share_random_minus_targeted_ci95":
                primary["exploit_share_random_minus_targeted_ci95"],
            "exploit_share_none": primary["exploit_share"]["none"],
            "exploit_share_random": primary["exploit_share"]["random"],
            "exploit_share_targeted": primary["exploit_share"]["targeted"],
        },
    }
