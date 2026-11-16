"""SIMULATED check: static leakiness audit versus exploit share among rewarded rollouts.

No model is involved. For each (rung, hack class) cell a rollout is, with probability ``prob``,
the class's probe mutation of the honest report (probes cycled per rollout) and the honest
report otherwise. A mutated rollout counts as an exploit only if the strict rung rejects it.
The exploit share is exploit rollouts the rung rewards (reward 1.0) over all rollouts it
rewards. The static acceptance share is the one of ``static_leakiness_audit``. The metric is
the Spearman rank correlation of the two across all cells.
"""
from __future__ import annotations

import random
from collections.abc import Sequence

from .fuzz_candidates import MUTATIONS
from .ladder import VERSIONS, grade, honest_policy
from .static_leakiness_audit import PROBES, spearman
from .tasks import ChunkIndex, Task


def run_leakiness_vs_hack_rate(
    tasks: Sequence[Task], index: ChunkIndex, *, group_size: int = 8,
    groups_per_task: int = 20, prob: float = 0.5, seed: int = 7,
) -> dict:
    if group_size < 2 or groups_per_task < 1:
        raise ValueError("group_size must be >= 2 and groups_per_task >= 1")
    if not 0.0 <= prob <= 1.0:
        raise ValueError("prob must be in [0, 1]")
    rng = random.Random(seed)
    honest = [honest_policy(t, index) for t in tasks]
    n_roll = group_size * groups_per_task
    static: dict[str, float] = {}
    hack: dict[str, float] = {}
    for m in MUTATIONS:
        probes = PROBES[m]
        outs = [[p(t, h) for t, h in zip(tasks, honest, strict=True)] for p in probes]
        bad = [[grade("strict", t, o) < 1.0 for t, o in zip(tasks, col, strict=True)]
               for col in outs]
        for v in VERSIONS:
            acc = [int(bad[k][i] and grade(v, tasks[i], outs[k][i]) >= 1.0)
                   for k in range(len(probes)) for i in range(len(tasks))]
            static[f"{v}/{m}"] = sum(acc) / len(acc) if acc else 0.0
            ex_rew = tot_rew = 0
            for i, t in enumerate(tasks):
                h_ok = grade(v, t, honest[i]) >= 1.0
                for r in range(n_roll):
                    if rng.random() < prob:
                        k = r % len(probes)
                        ok = grade(v, t, outs[k][i]) >= 1.0
                        tot_rew += ok
                        ex_rew += ok and bad[k][i]
                    else:
                        tot_rew += h_ok
            hack[f"{v}/{m}"] = ex_rew / tot_rew if tot_rew else 0.0
    cells = [f"{v}/{m}" for v in VERSIONS for m in MUTATIONS]
    rho = spearman([static[c] for c in cells], [hack[c] for c in cells])
    return {
        "n_tasks": len(tasks), "group_size": group_size, "groups_per_task": groups_per_task,
        "exploit_prob": prob, "cells": cells,
        "static_acceptance_share": static, "exploit_share_among_rewarded": hack,
        "leakiness": {"spearman_vs_hack_rate": rho, "n_cells": len(cells)},
    }
