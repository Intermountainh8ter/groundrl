"""SIMULATED check: verifier error direction versus the exploit share of reinforced rollouts.

No model is involved. A rollout is the exploit policy with probability ``p`` and the honest
policy otherwise. The grader reward comes from ``ladder.grade``. One error direction is injected
at a time, at the same rate ``e``: false positive (an exploit rollout the rung rejects is
accepted at 1.0) or false negative (an honest rollout the rung accepts is rejected at 0.0). A
rollout is reinforced when its observed reward is 1.0.
"""
from __future__ import annotations

import random
from collections.abc import Sequence

import numpy as np

from .group_degeneracy import p_key
from .ladder import VERSIONS, exploit_policy, grade, honest_policy
from .tasks import ChunkIndex, Task

DIRECTIONS = ("fp", "fn")


def _rollout(rng: random.Random, rh: float, re_: float, p: float, e: float, direction: str):
    """Return ``(exploit, reinforced)`` for one rollout."""
    exploit = rng.random() < p
    r = re_ if exploit else rh
    if direction == "fp" and exploit and r != 1.0 and rng.random() < e:
        r = 1.0
    elif direction == "fn" and not exploit and r == 1.0 and rng.random() < e:
        r = 0.0
    return exploit, r == 1.0


def _group_stats(rng, rh, re_, p, e, direction, g, n_groups):
    """Per-task counts: exploit-reinforced, reinforced, zero-variance groups."""
    ex_r = reinf = zero = 0
    for _ in range(n_groups):
        flags = [_rollout(rng, rh, re_, p, e, direction) for _ in range(g)]
        k = sum(r for _, r in flags)
        reinf += k
        ex_r += sum(x and r for x, r in flags)
        zero += k in (0, g)
    return ex_r, reinf, zero


def _share(ex: np.ndarray, rf: np.ndarray) -> float:
    s = rf.sum()
    return float(ex.sum() / s) if s > 0 else 0.0


def run_verifier_error_asymmetry(
    tasks: Sequence[Task],
    index: ChunkIndex,
    *,
    group_size: int = 8,
    groups_per_task: int = 20,
    probs: Sequence[float] = (0.1, 0.5, 0.9),
    error_rate: float = 0.2,
    seed: int = 7,
    n_boot: int = 1000,
    primary_version: str = "strict",
    primary_prob: float = 0.5,
) -> dict:
    if group_size < 2 or groups_per_task < 1:
        raise ValueError("group_size must be >= 2 and groups_per_task >= 1")
    if not 0.0 <= error_rate <= 1.0:
        raise ValueError("error_rate must be in [0, 1]")
    rng = random.Random(seed)
    boot_rng = np.random.default_rng(seed)
    table = []
    for t in tasks:
        h, e = honest_policy(t, index), exploit_policy(t, index)
        table.append({v: (grade(v, t, h), grade(v, t, e)) for v in VERSIONS})
    cells: dict[str, dict[str, dict]] = {}
    for p in probs:
        cells[p_key(p)] = {}
        for v in VERSIONS:
            arrs = {}
            for d in DIRECTIONS:
                rows = [_group_stats(rng, rewards[v][0], rewards[v][1], p, error_rate, d,
                                     group_size, groups_per_task) for rewards in table]
                arrs[d] = np.array(rows, dtype=float).reshape(-1, 3)
            n = len(table)
            idx = boot_rng.integers(0, max(n, 1), size=(n_boot, max(n, 1))) if n else None
            cell: dict = {}
            for d in DIRECTIONS:
                a = arrs[d]
                cell[d] = {
                    "exploit_share": _share(a[:, 0], a[:, 1]),
                    "zero_variance_group_share": float(a[:, 2].sum() / (n * groups_per_task))
                    if n else 0.0,
                    "reinforced_rollouts": int(a[:, 1].sum()),
                }
            diff = cell["fp"]["exploit_share"] - cell["fn"]["exploit_share"]
            lo = hi = diff
            if n:
                boots = np.array([
                    _share(arrs["fp"][r, 0], arrs["fp"][r, 1])
                    - _share(arrs["fn"][r, 0], arrs["fn"][r, 1]) for r in idx])
                lo, hi = (float(x) for x in np.percentile(boots, [2.5, 97.5]))
            cell["exploit_share_fp_minus_fn"] = diff
            cell["exploit_share_fp_minus_fn_ci95"] = [lo, hi]
            cells[p_key(p)][v] = cell
    primary = cells[p_key(primary_prob)][primary_version]
    return {
        "n_tasks": len(tasks),
        "group_size": group_size,
        "groups_per_task": groups_per_task,
        "error_rate": error_rate,
        "versions": list(VERSIONS),
        "exploit_probs": list(probs),
        "cells": cells,
        "asymmetry": {
            "primary_cell": {"version": primary_version, "exploit_prob": primary_prob},
            "exploit_share_fp_minus_fn": primary["exploit_share_fp_minus_fn"],
            "exploit_share_fp_minus_fn_ci95": primary["exploit_share_fp_minus_fn_ci95"],
        },
    }
