"""SIMULATED GRPO groups: does the mean hack-rollout advantage change sign between rewards?

No model is involved. Same group sampler as ``degenerate_group_by_reward``. Advantage is
(r - mean) / (std + 1e-6) within the group. Groups with no hack rollout are skipped.
"""
from __future__ import annotations

import random
from collections.abc import Sequence

from .audit import POLICY_BY_NAME, run_policy
from .degenerate_group_by_reward import HACKS
from .example import Example
from .rewards import score_completion

TOL = 1e-9
PAIRS = (("v0", "v2"), ("v0", "v1"), ("v1", "v2"))


def _sign(x: float) -> int:
    return 1 if x > TOL else -1 if x < -TOL else 0


def _hack_adv_sign(picks: Sequence[bool], ro: float, rh: float) -> int:
    rs = [rh if x else ro for x in picks]
    m = sum(rs) / len(rs)
    sd = (sum((r - m) ** 2 for r in rs) / len(rs)) ** 0.5
    adv = (rh - m) / (sd + 1e-6)
    return _sign(adv)


def run_sign_flip_by_reward(
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
    per_hack: dict[str, dict] = {}
    for hack in hacks:
        spec = POLICY_BY_NAME[hack]
        rng = random.Random(f"{seed}|{hack}")
        n = 0
        flips = {p: 0 for p in PAIRS}
        strict = {p: 0 for p in PAIRS}
        for e in examples:
            o, h = run_policy(oracle, e, seed), run_policy(spec, e, seed)
            rw = {v: (score_completion(v, o, e)[0], score_completion(v, h, e)[0])
                  for v in ("v0", "v1", "v2")}
            for _ in range(groups_per_task):
                picks = [rng.random() < prob for _ in range(group_size)]
                if not any(picks):
                    continue
                n += 1
                sg = {v: _hack_adv_sign(picks, *rw[v]) for v in rw}
                for a, b in PAIRS:
                    if sg[a] != sg[b]:
                        flips[(a, b)] += 1
                    if sg[a] * sg[b] == -1:
                        strict[(a, b)] += 1
        per_hack[hack] = {"n_groups": n,
                          "flip": {f"{a}_vs_{b}": (flips[(a, b)] / n if n else 0.0)
                                   for a, b in PAIRS},
                          "strict": {f"{a}_vs_{b}": (strict[(a, b)] / n if n else 0.0)
                                     for a, b in PAIRS}}
    out = {}
    for a, b in PAIRS:
        k = f"{a}_vs_{b}"
        out[k] = {
            "mean_share": sum(per_hack[h]["flip"][k] for h in hacks) / len(hacks),
            "mean_strict_reversal_share": sum(per_hack[h]["strict"][k] for h in hacks) / len(hacks),
            "per_hack": {h: per_hack[h]["flip"][k] for h in hacks},
            "per_hack_strict": {h: per_hack[h]["strict"][k] for h in hacks},
        }
    return {
        "n_tasks": len(examples),
        "group_size": group_size,
        "groups_per_task": groups_per_task,
        "exploit_prob": prob,
        "hacks": list(hacks),
        "groups_with_hack_per_hack": {h: per_hack[h]["n_groups"] for h in hacks},
        "sign_flip_share": out,
    }
