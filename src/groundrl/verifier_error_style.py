"""SIMULATED check: style-keyed versus independent verifier false-positive error.

No model is involved. An exploit rollout carries a style feature with probability ``s``. In
``style`` mode a rejected exploit is accepted with probability ``acc_styled`` if styled and
``acc_plain`` otherwise; in ``independent`` mode with the flat ``error_rate``. At the base
style rate the style marginal equals ``error_rate`` by construction.
"""
from __future__ import annotations

import random
from collections.abc import Sequence

from .ladder import exploit_policy, grade, honest_policy
from .tasks import ChunkIndex, Task


def _arm(rewards, rng, p, g, groups, style_rate, accept_prob):
    """Return ``(exploit_reinforced, reinforced)`` over all tasks for one arm."""
    ex = rf = 0
    for rh, re_ in rewards:
        for _ in range(groups):
            for _ in range(g):
                exploit = rng.random() < p
                r = re_ if exploit else rh
                if exploit and r != 1.0 and rng.random() < accept_prob(
                        rng.random() < style_rate):
                    r = 1.0
                if r == 1.0:
                    rf += 1
                    ex += exploit
    return ex, rf


def run_verifier_error_style(
    tasks: Sequence[Task],
    index: ChunkIndex,
    *,
    group_size: int = 8,
    groups_per_task: int = 20,
    error_rate: float = 0.2,
    prob: float = 0.5,
    base_style: float = 0.2,
    raised_style: float = 0.8,
    acc_styled: float = 0.8,
    acc_plain: float = 0.05,
    seed: int = 7,
    version: str = "strict",
) -> dict:
    if group_size < 2 or groups_per_task < 1:
        raise ValueError("group_size must be >= 2 and groups_per_task >= 1")
    for v in (error_rate, base_style, raised_style, acc_styled, acc_plain):
        if not 0.0 <= v <= 1.0:
            raise ValueError("rates must be in [0, 1]")
    rng = random.Random(seed)
    rewards = []
    for t in tasks:
        h, e = honest_policy(t, index), exploit_policy(t, index)
        rewards.append((grade(version, t, h), grade(version, t, e)))

    def keyed(styled):
        return acc_styled if styled else acc_plain

    arms = {
        "independent": (base_style, lambda _s: error_rate),
        "style_base": (base_style, keyed),
        "style_raised": (raised_style, keyed),
    }
    out = {}
    for name, (rate, fn) in arms.items():
        ex, rf = _arm(rewards, rng, prob, group_size, groups_per_task, rate, fn)
        out[name] = {"exploit_share": ex / rf if rf else 0.0, "reinforced_rollouts": rf}
    ind = out["independent"]["exploit_share"]
    return {
        "n_tasks": len(tasks),
        "group_size": group_size,
        "groups_per_task": groups_per_task,
        "error_rate": error_rate,
        "exploit_prob": prob,
        "base_style": base_style,
        "raised_style": raised_style,
        "version": version,
        "arms": out,
        "base_diff": out["style_base"]["exploit_share"] - ind,
        "raised_diff": out["style_raised"]["exploit_share"] - ind,
    }
