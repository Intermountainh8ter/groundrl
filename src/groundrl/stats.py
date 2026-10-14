"""Small statistics helpers (numpy only): bootstrap confidence intervals."""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np


def bootstrap_ci(
    values: Sequence[float], stat: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 2000, alpha: float = 0.05, seed: int = 0,
) -> tuple[float, float, float]:
    """``(point estimate, lower, upper)`` of ``stat`` with a percentile bootstrap interval."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(stat(arr))
    if arr.size == 1:
        return point, point, point
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    if stat is np.mean:
        boots = arr[idx].mean(axis=1)
    else:
        boots = np.array([stat(arr[row]) for row in idx])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def mean_ci(values: Sequence[float], n_boot: int = 2000, seed: int = 0) -> dict[str, float]:
    """Mean with a 95% bootstrap interval, as a JSON-friendly dict."""
    point, lo, hi = bootstrap_ci(values, np.mean, n_boot=n_boot, seed=seed)
    return {"mean": point, "lo": lo, "hi": hi, "n": len(values)}
