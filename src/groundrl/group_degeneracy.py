"""SIMULATED GRPO group sampler over the strictness ladder: share of zero-variance groups.

No model is involved. Each rollout in a group is the scripted exploit policy with probability
``p`` and the honest policy otherwise; a group is degenerate for GRPO when every rollout gets the
same reward, so the group-relative advantage is zero.
"""
from __future__ import annotations

import random
from collections.abc import Sequence

from .ladder import VERSIONS, exploit_policy, grade, honest_policy
from .tasks import ChunkIndex, Task


def p_key(p: float) -> str:
    return f"p_{p:g}"


def run_group_degeneracy(
    tasks: Sequence[Task],
    index: ChunkIndex,
    *,
    group_size: int = 8,
    groups_per_task: int = 20,
    probs: Sequence[float] = (0.5,),
    seed: int = 7,
) -> dict:
    if group_size < 2 or groups_per_task < 1:
        raise ValueError("group_size must be >= 2 and groups_per_task >= 1")
    rng = random.Random(seed)
    # Rewards are deterministic per (task, policy, rung), so compute them once.
    table = []
    for t in tasks:
        h, e = honest_policy(t, index), exploit_policy(t, index)
        table.append({v: (grade(v, t, h), grade(v, t, e)) for v in VERSIONS})
    total = len(tasks) * groups_per_task
    share: dict[str, dict[str, float]] = {}
    for p in probs:
        zero = {v: 0 for v in VERSIONS}
        for rewards in table:
            for _ in range(groups_per_task):
                picks = [rng.random() < p for _ in range(group_size)]
                for v in VERSIONS:
                    rh, re_ = rewards[v]
                    if len({re_ if x else rh for x in picks}) == 1:
                        zero[v] += 1
        share[p_key(p)] = {v: (zero[v] / total if total else 0.0) for v in VERSIONS}
    return {
        "n_tasks": len(tasks),
        "group_size": group_size,
        "groups_per_task": groups_per_task,
        "versions": list(VERSIONS),
        "exploit_probs": list(probs),
        "zero_variance_group_share": share,
    }
