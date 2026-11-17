"""Isomorphic-perturbation rung: strict grading plus the task re-posed with gold items reversed.

SIMULATED, no model. A memorizer replays the original-order gold items; the honest policy answers
whichever task variant it is shown. ``strict_iso`` accepts a submission only if the strict rung
accepts it on the original task and on the reversed-order variant.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from .ladder import grade, honest_policy
from .tasks import ChunkIndex, Task

RUNGS = ("strict", "strict_iso")


def perturb(task: Task) -> Task:
    """The same task with its gold items in reversed order."""
    return replace(task, gold_tokens=list(reversed(task.gold_tokens)))


def memorizer_policy(task: Task, index: ChunkIndex) -> str:
    """Replays the original-order gold answer whatever variant is posed."""
    return honest_policy(task, index)


def _accepts(rung: str, task: Task, perturbed: Task, text: str) -> bool:
    ok = grade("strict", task, text) >= 1.0
    if rung == "strict_iso":
        ok = ok and grade("strict", perturbed, text) >= 1.0
    return ok


def run_isomorphic_rung(tasks: Sequence[Task], index: ChunkIndex) -> dict:
    n = len(tasks)
    if n == 0:
        raise ValueError("no tasks")
    mem = {r: 0 for r in RUNGS}
    hon = {r: 0 for r in RUNGS}
    survivors = {"single_item": 0, "unanswerable": 0, "palindromic_multi": 0}
    for t in tasks:
        p = perturb(t)
        m_text = memorizer_policy(t, index)
        for r in RUNGS:
            mem[r] += _accepts(r, t, p, m_text)
        # honest policy answers the variant it is shown: original for strict, both for iso
        h_orig, h_pert = honest_policy(t, index), honest_policy(p, index)
        hon["strict"] += grade("strict", t, h_orig) >= 1.0
        hon["strict_iso"] += (grade("strict", t, h_orig) >= 1.0
                              and grade("strict", p, h_pert) >= 1.0)
        if _accepts("strict_iso", t, p, m_text):
            if not t.answerable:
                survivors["unanswerable"] += 1
            elif len(t.gold_tokens) == 1:
                survivors["single_item"] += 1
            else:
                survivors["palindromic_multi"] += 1
    return {
        "n_tasks": n,
        "rungs": list(RUNGS),
        "memorizer_accept_rate": {r: mem[r] / n for r in RUNGS},
        "honest_accept_rate": {r: hon[r] / n for r in RUNGS},
        "memorizer_survivors_by_kind": {k: v / n for k, v in survivors.items()},
    }
