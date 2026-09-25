"""Statistics the audit needs: dependent-observation corrections, FDR, and the deflated Sharpe ratio."""

from __future__ import annotations

import numpy as np
from scipy import stats

EULER_GAMMA = 0.5772156649015329


def newey_west_t(x: np.ndarray, lag: int) -> float:
    """t-statistic of the mean with a Newey-West (Bartlett) HAC variance for `lag` autocorrelations."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    d = x - x.mean()
    var = d @ d / n
    for k in range(1, lag + 1):
        w = 1.0 - k / (lag + 1.0)
        var += 2.0 * w * (d[k:] @ d[:-k]) / n
    se = np.sqrt(var / n)
    return float(x.mean() / se) if se > 0 else 0.0


def block_bootstrap_se(x: np.ndarray, block: int, n_boot: int = 1000, seed: int = 0) -> float:
    """Standard error of the mean from a moving-block bootstrap; blocks preserve the dependence."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, max(n - block + 1, 1), size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :n]
    means = x[idx].mean(axis=1)
    return float(means.std(ddof=1))


def effective_sample_size(x: np.ndarray, lag: int) -> float:
    """N / (1 + 2 * sum of the first `lag` autocorrelations), floored at 1."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    d = x - x.mean()
    denom = d @ d
    if denom == 0:
        return float(n)
    rho = [(d[k:] @ d[:-k]) / denom for k in range(1, lag + 1)]
    return float(max(n / (1.0 + 2.0 * sum(rho)), 1.0))


def benjamini_hochberg(pvalues: np.ndarray, q: float = 0.05) -> np.ndarray:
    """Boolean mask of hypotheses rejected at false-discovery rate q."""
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order]
    thresholds = q * (np.arange(1, m + 1) / m)
    below = np.nonzero(ranked <= thresholds)[0]
    rejected = np.zeros(m, dtype=bool)
    if len(below):
        rejected[order[: below[-1] + 1]] = True
    return rejected


def expected_max_sharpe(n_trials: int, sharpe_var: float) -> float:
    """Expected maximum Sharpe among n_trials of pure noise (Bailey & Lopez de Prado, 2014)."""
    if n_trials < 2 or sharpe_var <= 0:
        return 0.0
    z = stats.norm.ppf
    return float(np.sqrt(sharpe_var) * ((1 - EULER_GAMMA) * z(1 - 1 / n_trials) + EULER_GAMMA * z(1 - 1 / (n_trials * np.e))))


def deflated_sharpe(sharpe_per_period: float, n_periods: int, skew: float, kurt: float, n_trials: int, sharpe_var: float) -> float:
    """Probability that the observed (non-annualized) Sharpe beats the best of n_trials noise strategies."""
    sr0 = expected_max_sharpe(n_trials, sharpe_var)
    denom = np.sqrt(max(1.0 - skew * sharpe_per_period + (kurt - 1.0) / 4.0 * sharpe_per_period**2, 1e-12))
    z = (sharpe_per_period - sr0) * np.sqrt(n_periods - 1) / denom
    return float(stats.norm.cdf(z))
