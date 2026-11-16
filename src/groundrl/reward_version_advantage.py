"""SIMULATED GRPO groups mixing the oracle with an audit hack, under the real reward versions.

No model is involved. Each rollout is the scripted hack policy with probability ``prob`` and the
oracle otherwise; rewards come from ``score_completion`` (v0, v1, v2), not the ladder rungs. A
group is hack-positive when it holds at least one hack rollout and their mean group-normalised
advantage is strictly positive.
"""
from __future__ import annotations

import random
from collections.abc import Sequence

from .audit import POLICY_BY_NAME, run_policy
from .example import Example
from .exploit_advantage import _EPS, exploit_mean_advantage
from .rewards import score_completion

VERSIONS = ("v0", "v1", "v2")
HACKS = ("copy_chunk", "cite_everything", "stuff_numbers", "citation_spam", "wrong_chunk",
         "one_wrong_constant", "paraphrase_no_tokens")


def run_reward_version_advantage(
    examples: Sequence[Example],
    *,
    hacks: Sequence[str] = HACKS,
    group_size: int = 8,
    groups_per_task: int = 20,
    prob: float = 0.5,
    seed: int = 7,
) -> dict:
    if group_size < 2 or groups_per_task < 1:
        raise ValueError("group_size must be >= 2 and groups_per_task >= 1")
    if not hacks:
        raise ValueError("hacks must be non-empty")
    oracle = POLICY_BY_NAME["oracle"]
    total = len(examples) * groups_per_task
    per_hack: dict[str, dict[str, float]] = {}
    for hack in hacks:
        spec = POLICY_BY_NAME[hack]
        rng = random.Random(f"{seed}|{hack}")
        rows = []
        for e in examples:
            o, h = run_policy(oracle, e, seed), run_policy(spec, e, seed)
            rows.append({v: (score_completion(v, o, e)[0], score_completion(v, h, e)[0])
                         for v in VERSIONS})
        pos = {v: 0 for v in VERSIONS}
        for rewards in rows:
            for _ in range(groups_per_task):
                picks = [rng.random() < prob for _ in range(group_size)]
                for v in VERSIONS:
                    ro, rh = rewards[v]
                    m = exploit_mean_advantage([rh if x else ro for x in picks], picks)
                    if m is not None and m > _EPS:
                        pos[v] += 1
        per_hack[hack] = {v: (pos[v] / total if total else 0.0) for v in VERSIONS}
    by_reward = {
        v: {"mean_positive_share": sum(per_hack[h][v] for h in hacks) / len(hacks),
            "per_hack": {h: per_hack[h][v] for h in hacks}}
        for v in VERSIONS
    }
    return {
        "n_tasks": len(examples),
        "group_size": group_size,
        "groups_per_task": groups_per_task,
        "exploit_prob": prob,
        "hacks": list(hacks),
        "versions": list(VERSIONS),
        "exploit_advantage_by_reward": by_reward,
    }
