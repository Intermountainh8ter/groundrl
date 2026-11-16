"""SIMULATED check: task-clustered versus independent verifier false-positive error.

No model is involved. A rollout is the exploit policy with probability ``p`` and the honest
policy otherwise. In ``independent`` mode a rejected exploit rollout is accepted with
probability ``e``. In ``clustered`` mode a fixed share ``e`` of tasks are blind spots (chosen
by the seeded rng) where every exploit rollout is accepted. The marginal rate is the same.
"""
from __future__ import annotations

import random
from collections.abc import Sequence

from .ladder import exploit_policy, grade, honest_policy
from .tasks import ChunkIndex, Task

MODES = ("independent", "clustered")


def blind_spots(rng: random.Random, n: int, rate: float) -> list[bool]:
    """Mark exactly round(rate * n) of n tasks as blind spots."""
    k = round(rate * n)
    chosen = set(rng.sample(range(n), k))
    return [i in chosen for i in range(n)]


def _group(rng, rh, re_, p, g, accept):
    """Return ``(exploit_reinforced, reinforced)`` for one group.

    ``accept`` is a callable returning True when a rejected exploit is accepted.
    """
    ex_r = reinf = 0
    for _ in range(g):
        exploit = rng.random() < p
        r = re_ if exploit else rh
        if exploit and r != 1.0 and accept():
            r = 1.0
        if r == 1.0:
            reinf += 1
            ex_r += exploit
    return ex_r, reinf


def run_verifier_error_clustering(
    tasks: Sequence[Task],
    index: ChunkIndex,
    *,
    group_size: int = 8,
    groups_per_task: int = 20,
    error_rate: float = 0.2,
    prob: float = 0.5,
    seed: int = 7,
    version: str = "strict",
) -> dict:
    if group_size < 2 or groups_per_task < 1:
        raise ValueError("group_size must be >= 2 and groups_per_task >= 1")
    if not 0.0 <= error_rate <= 1.0:
        raise ValueError("error_rate must be in [0, 1]")
    rng = random.Random(seed)
    rewards = []
    for t in tasks:
        h, e = honest_policy(t, index), exploit_policy(t, index)
        rewards.append((grade(version, t, h), grade(version, t, e)))
    spots = blind_spots(rng, len(tasks), error_rate)
    out: dict[str, dict] = {}
    for mode in MODES:
        ex = rf = zero = 0
        for (rh, re_), spot in zip(rewards, spots, strict=True):
            if mode == "clustered":
                def accept(spot=spot):
                    return spot
            else:
                def accept():
                    return rng.random() < error_rate
            for _ in range(groups_per_task):
                a, b = _group(rng, rh, re_, prob, group_size, accept)
                ex += a
                rf += b
                zero += b in (0, group_size)
        n_groups = len(tasks) * groups_per_task
        out[mode] = {
            "exploit_share": ex / rf if rf else 0.0,
            "zero_variance_group_share": zero / n_groups if n_groups else 0.0,
            "reinforced_rollouts": rf,
        }
    return {
        "n_tasks": len(tasks),
        "n_blind_spots": sum(spots),
        "group_size": group_size,
        "groups_per_task": groups_per_task,
        "error_rate": error_rate,
        "exploit_prob": prob,
        "version": version,
        "modes": out,
        "zero_variance_diff": out["clustered"]["zero_variance_group_share"]
        - out["independent"]["zero_variance_group_share"],
        "exploit_share_diff": out["clustered"]["exploit_share"]
        - out["independent"]["exploit_share"],
    }
