"""SIMULATED GRPO groups: independent versus shared-seam verifier false positives.

No model is involved. Each rollout is the honest policy (strict reward ``r_ok``) with probability
``q`` and the corrupted honest policy (strict reward ``r_bad``) otherwise. A flawed verifier grants
``r_ok`` to a wrong rollout as a false positive. Two error models share the same marginal
false-positive rate ``fp``: ``independent`` flips each wrong rollout on its own; ``shared_seam``
flips every wrong rollout in a group together (one draw per group, standing for a common grader
seam). A group *sign flips* when the mean group-normalised advantage of its wrong rollouts is
strictly negative under the strict grader but strictly positive under the flawed one.
"""
from __future__ import annotations

import random
from collections.abc import Sequence

import numpy as np

from .exploit_advantage import corrupted_honest_policy
from .ladder import grade, honest_policy
from .stats import bootstrap_ci
from .tasks import ChunkIndex, Task

_EPS = 1e-9
MODES = ("independent", "shared_seam")


def _adv(rewards: Sequence[float]) -> list[float]:
    n = len(rewards)
    mean = sum(rewards) / n
    std = (sum((r - mean) ** 2 for r in rewards) / n) ** 0.5
    return [0.0 if std < _EPS else (r - mean) / std for r in rewards]


def _wrong_mean_adv(rewards: Sequence[float], wrong: Sequence[bool]) -> float | None:
    adv = [a for a, w in zip(_adv(rewards), wrong, strict=True) if w]
    return sum(adv) / len(adv) if adv else None


def group_outcome(
    r_ok: float, r_bad: float, ok: Sequence[bool], fp: Sequence[bool]
) -> tuple[bool, bool]:
    """``(sign_flip, degenerate)`` for one group given per-rollout ok and false-positive flags."""
    wrong = [not o for o in ok]
    strict = [r_ok if o else r_bad for o in ok]
    noisy = [r_ok if (o or f) else r_bad for o, f in zip(ok, fp, strict=True)]
    ms, mn = _wrong_mean_adv(strict, wrong), _wrong_mean_adv(noisy, wrong)
    flip = ms is not None and mn is not None and ms < -_EPS and mn > _EPS
    return flip, max(noisy) - min(noisy) < _EPS


def run_seam_errors(
    tasks: Sequence[Task],
    index: ChunkIndex,
    *,
    group_size: int = 8,
    groups_per_task: int = 20,
    fp_rate: float = 0.10,
    honest_prob: float = 0.5,
    seed: int = 7,
    n_boot: int = 1000,
) -> dict:
    if group_size < 2 or groups_per_task < 1:
        raise ValueError("group_size must be >= 2 and groups_per_task >= 1")
    if not 0.0 <= fp_rate <= 1.0 or not 0.0 <= honest_prob <= 1.0:
        raise ValueError("fp_rate and honest_prob must be in [0, 1]")
    rng = random.Random(seed)
    flips = {m: [] for m in MODES}
    degen = {m: [] for m in MODES}
    for t in tasks:
        r_ok = grade("strict", t, honest_policy(t, index))
        r_bad = grade("strict", t, corrupted_honest_policy(t, index))
        cnt = {m: [0, 0] for m in MODES}
        for _ in range(groups_per_task):
            ok = [rng.random() < honest_prob for _ in range(group_size)]
            fps = {
                "independent": [rng.random() < fp_rate for _ in range(group_size)],
                "shared_seam": [rng.random() < fp_rate] * group_size,
            }
            for m in MODES:
                f, d = group_outcome(r_ok, r_bad, ok, fps[m])
                cnt[m][0] += f
                cnt[m][1] += d
        for m in MODES:
            flips[m].append(cnt[m][0] / groups_per_task)
            degen[m].append(cnt[m][1] / groups_per_task)

    def mean(xs):
        return float(np.mean(xs)) if len(xs) else 0.0

    def diff_ci(d):
        diff = np.asarray(d["shared_seam"]) - np.asarray(d["independent"])
        p, lo, hi = bootstrap_ci(diff, n_boot=n_boot, seed=seed)
        return {"point": p, "lo": lo, "hi": hi}

    return {
        "n_tasks": len(tasks),
        "group_size": group_size,
        "groups_per_task": groups_per_task,
        "fp_rate": fp_rate,
        "honest_prob": honest_prob,
        "advantage_sign_flip_share": {m: mean(flips[m]) for m in MODES},
        "degenerate_group_share": {m: mean(degen[m]) for m in MODES},
        "sign_flip_difference_ci": diff_ci(flips),
        "degenerate_difference_ci": diff_ci(degen),
    }
