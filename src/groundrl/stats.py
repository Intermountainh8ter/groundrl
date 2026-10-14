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


def ratio_ci(
    numerator: Sequence[float], denominator: Sequence[float], n_boot: int = 2000,
    alpha: float = 0.05, seed: int = 0,
) -> dict[str, float | int | None]:
    """Ratio of sums (a precision or recall) with a bootstrap interval over tasks.

    Returns ``value`` ``None`` when the denominator is zero (the metric is undefined, for example
    abstention precision when the model never abstains).
    """
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    total = float(den.sum())
    if num.size == 0 or total == 0.0:
        return {"value": None, "lo": None, "hi": None, "n": int(num.size)}
    point = float(num.sum() / total)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, num.size, size=(n_boot, num.size))
    boot_num, boot_den = num[idx].sum(axis=1), den[idx].sum(axis=1)
    ok = boot_den > 0
    if not ok.any():  # pragma: no cover - requires a degenerate resample of a non-empty set
        return {"value": point, "lo": None, "hi": None, "n": int(num.size)}
    ratios = boot_num[ok] / boot_den[ok]
    lo, hi = np.percentile(ratios, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"value": point, "lo": float(lo), "hi": float(hi), "n": int(num.size)}
