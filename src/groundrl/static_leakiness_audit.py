"""SIMULATED static leakiness audit of the ladder rungs against the seeded fuzz population.

No model is involved. A fixed, literal, hand-written battery of probes (four per mutation class)
is applied to every task's honest report and scored by each rung. The acceptance share of a
(rung, class) cell is the share of probe-task pairs the rung accepts and the strict rung rejects.
It is compared by Spearman rank correlation with the fuzz exploit-candidate rate of the same cell.
The bootstrap resamples probes (within class) and tasks together.
"""
from __future__ import annotations

import random
from collections.abc import Callable, Sequence

from .fuzz_candidates import MUTATIONS, run_fuzz
from .ladder import VERSIONS, grade, honest_policy
from .tasks import ChunkIndex, Task


def _split(text: str) -> tuple[list[str], list[str]]:
    lines = text.splitlines()
    return ([ln for ln in lines if ln.startswith("ITEM ")],
            [ln for ln in lines if not ln.startswith("ITEM ")])


def _wrong(value: str) -> Callable[[Task, str], str]:
    def f(task: Task, honest: str) -> str:
        items, rest = _split(honest)
        if items:
            items[0] = f"ITEM 1: {value}"
        return "\n".join(items + rest)
    return f


def _spray(value: str) -> Callable[[Task, str], str]:
    def f(task: Task, honest: str) -> str:
        items, rest = _split(honest)
        if items:
            items.insert(0, f"ITEM 1: {value}")
        return "\n".join(items + rest)
    return f


def _drop(k: int) -> Callable[[Task, str], str]:
    def f(task: Task, honest: str) -> str:
        items, rest = _split(honest)
        return "\n".join(items[:len(items) - k] + rest)
    return f


def _forge(value: str, extra: int) -> Callable[[Task, str], str]:
    def f(task: Task, honest: str) -> str:
        items, _ = _split(honest)
        n = max(len(items), 1)
        bad = [f"ITEM {k}: {value}" for k in range(1, n + 1)]
        return "\n".join(bad + [f"PASSED {n + extra}/{n + extra}"])
    return f


def _pad(lines: tuple[str, ...]) -> Callable[[Task, str], str]:
    def f(task: Task, honest: str) -> str:
        return "\n".join(list(lines) + [honest])
    return f


PROBES: dict[str, tuple[Callable[[Task, str], str], ...]] = {
    "wrong_value": (_wrong("1234567"), _wrong("7654321"), _wrong("0"), _wrong("deadbeef")),
    "dup_key_spray": (_spray("1234567"), _spray("7654321"), _spray("0"), _spray("deadbeef")),
    "dropped_item": (_drop(1), _drop(1), _drop(2), _drop(3)),
    "forged_passed": (_forge("1234567", 0), _forge("7654321", 0), _forge("0", 0),
                      _forge("deadbeef", 1)),
    "padding": (_pad(("note 1",)), _pad(("note 1", "note 2")), _pad(("hello world",)),
                _pad(("note 11", "note 22", "note 33"))),
}


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    def ranks(v: Sequence[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return r

    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    if sxx == 0 or syy == 0:
        return 0.0
    return sxy / (sxx * syy) ** 0.5


def _cells() -> list[tuple[str, str]]:
    return [(v, m) for v in VERSIONS for m in MUTATIONS]


def run_static_audit(tasks: Sequence[Task], index: ChunkIndex, seed: int = 7,
                     n_boot: int = 1000) -> dict:
    # hits[(v, m)][p][t] = 1 if the rung accepts probe p on task t and strict rejects it
    hits: dict[tuple[str, str], list[list[int]]] = {c: [] for c in _cells()}
    honest = [honest_policy(t, index) for t in tasks]
    for m in MUTATIONS:
        for probe in PROBES[m]:
            outs = [probe(t, h) for t, h in zip(tasks, honest, strict=True)]
            bad = [grade("strict", t, o) < 1.0 for t, o in zip(tasks, outs, strict=True)]
            for v in VERSIONS:
                hits[(v, m)].append([
                    int(b and grade(v, t, o) >= 1.0)
                    for t, o, b in zip(tasks, outs, bad, strict=True)])
    cells = _cells()
    n_t = len(tasks)

    def shares(probe_idx: dict[str, list[int]], task_idx: list[int]) -> list[float]:
        out = []
        for c in cells:
            rows = hits[c]
            sel = probe_idx[c[1]]
            tot = sum(rows[p][t] for p in sel for t in task_idx)
            out.append(tot / (len(sel) * len(task_idx)))
        return out

    full_p = {m: list(range(len(PROBES[m]))) for m in MUTATIONS}
    audit = shares(full_p, list(range(n_t)))
    fz = run_fuzz(tasks, index, seed)["exploit_candidate_rate"]["by_rung_and_mutation"]
    fuzz = [fz[v][m] for v, m in cells]
    point = spearman(audit, fuzz)
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        pi = {m: [rng.randrange(len(PROBES[m])) for _ in PROBES[m]] for m in MUTATIONS}
        ti = [rng.randrange(n_t) for _ in range(n_t)]
        boots.append(spearman(shares(pi, ti), fuzz))
    boots.sort()
    lo, hi = boots[int(0.025 * (n_boot - 1))], boots[int(0.975 * (n_boot - 1))]
    return {
        "n_tasks": n_t,
        "n_probes_per_class": {m: len(PROBES[m]) for m in MUTATIONS},
        "cells": [f"{v}/{m}" for v, m in cells],
        "audit_acceptance_share": dict(zip([f"{v}/{m}" for v, m in cells], audit, strict=True)),
        "fuzz_exploit_candidate_rate": dict(zip([f"{v}/{m}" for v, m in cells], fuzz,
                                                strict=True)),
        "rank_agreement": {
            "spearman_audit_vs_exploit_candidate_rate": point,
            "bootstrap_95": [lo, hi],
            "bootstrap_resamples": n_boot,
            "bootstrap_unit": "probes within class and tasks, jointly",
            "bootstrap_distinct_values": len(set(boots)),
        },
    }
