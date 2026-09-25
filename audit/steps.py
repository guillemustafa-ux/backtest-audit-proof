"""The nine audit steps. Each takes the naive setup, removes one artifact, and returns the numbers for the report."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from audit import stats as st
from audit.engine import Params, Rules, Trade, _walk, atr, breakout_signals, daily_returns, run, sharpe, summary

NAIVE = Rules(lookahead=True, same_bar_best_case=True, cost_bps=0.0)
HONEST = Rules(lookahead=False, same_bar_best_case=False, cost_bps=0.0)


def small_grid() -> list[Params]:
    return [Params(lookback=lb, target_atr=t, stop_atr=s) for lb in (5, 20, 60) for t in (1.0, 3.0) for s in (0.5, 2.0)]


def full_grid() -> list[Params]:
    from naive_backtest import GRID
    return GRID


# --- 1. look-ahead / leakage -------------------------------------------------------------------

def lookahead(prices: pd.DataFrame, params: Params) -> dict:
    before = summary(prices, run(prices, params, NAIVE))
    after_rules = Rules(lookahead=False, same_bar_best_case=True, cost_bps=0.0)
    after = summary(prices, run(prices, params, after_rules))
    return {"before": before, "after": after}


# --- 2. same-bar ambiguity ---------------------------------------------------------------------

def same_bar(prices: pd.DataFrame, params: Params) -> dict:
    best = Rules(lookahead=False, same_bar_best_case=True)
    worst = Rules(lookahead=False, same_bar_best_case=False)
    t_best, t_worst = run(prices, params, best), run(prices, params, worst)
    high, low, a = prices["high"].to_numpy(), prices["low"].to_numpy(), atr(prices)
    ambiguous = 0
    for t in t_best:
        target, stop = t.entry_price + params.target_atr * a[t.entry_idx], t.entry_price - params.stop_atr * a[t.entry_idx]
        if high[t.exit_idx] >= target and low[t.exit_idx] <= stop:
            ambiguous += 1
    return {"before": summary(prices, t_best), "after": summary(prices, t_worst), "ambiguous_bars": ambiguous, "trades": len(t_best)}


# --- 3. transaction costs ----------------------------------------------------------------------

def costs(prices: pd.DataFrame, params: Params, cost_bps: float = 20.0) -> dict:
    free = run(prices, params, HONEST)
    paid_rules = Rules(lookahead=False, same_bar_best_case=False, cost_bps=cost_bps)
    paid = run(prices, params, paid_rules)
    return {"before": summary(prices, free), "after": summary(prices, paid, cost_bps), "cost_bps": cost_bps}


# --- 4. overlapping forward-return windows / effective sample size -----------------------------

def _forward_returns(prices: pd.DataFrame, horizon: int) -> np.ndarray:
    close = prices["close"].to_numpy()
    fwd = np.full(len(close), np.nan)
    fwd[: -horizon] = close[horizon:] / close[:-horizon] - 1.0
    return fwd


def overlapping_windows(prices: pd.DataFrame, params: Params) -> dict:
    """Signal-bar forward returns overlap when signals cluster; the naive t-test counts each as independent."""
    horizon = params.max_hold
    signals = np.concatenate([[False], breakout_signals(prices, params.lookback)[:-1]])
    fwd = _forward_returns(prices, horizon)
    x = fwd[signals & ~np.isnan(fwd)]
    t_naive, _ = stats.ttest_1samp(x, 0.0)
    n_eff = st.effective_sample_size(x, lag=horizon - 1)
    t_nw = st.newey_west_t(x, lag=horizon - 1)
    se_boot = st.block_bootstrap_se(x, block=horizon, n_boot=2000, seed=0)
    return {
        "horizon": horizon, "raw_n": int(len(x)), "effective_n": n_eff,
        "t_naive": float(t_naive), "t_corrected": t_nw, "t_bootstrap": float(x.mean() / se_boot) if se_boot > 0 else 0.0,
        "mean_fwd": float(x.mean()),
    }


# --- 5. event vs baseline dependence -----------------------------------------------------------

def event_vs_baseline(prices: pd.DataFrame, params: Params) -> dict:
    """Testing signal returns against zero ignores what every bar does; the baseline is the unconditional forward return."""
    horizon = params.max_hold
    signals = np.concatenate([[False], breakout_signals(prices, params.lookback)[:-1]])
    fwd = _forward_returns(prices, horizon)
    valid = ~np.isnan(fwd)
    event, base = fwd[signals & valid], fwd[valid]
    t_vs_zero, _ = stats.ttest_1samp(event, 0.0)
    t_vs_base, p_vs_base = stats.ttest_ind(event, base[~signals[valid]], equal_var=False)
    return {
        "event_mean": float(event.mean()), "baseline_mean": float(base.mean()), "excess_mean": float(event.mean() - base.mean()),
        "t_vs_zero": float(t_vs_zero), "t_vs_baseline": float(t_vs_base), "p_vs_baseline": float(p_vs_base),
    }


# --- 6. directional vs two-sided ---------------------------------------------------------------

def two_sided(prices: pd.DataFrame, params: Params) -> dict:
    d = daily_returns(prices, run(prices, params, HONEST))
    d = d[d != 0]
    t, p_two = stats.ttest_1samp(d, 0.0)
    p_one = p_two / 2 if t > 0 else 1 - p_two / 2
    return {"t": float(t), "p_one_sided": float(p_one), "p_two_sided": float(p_two), "n": int(len(d))}


# --- 7. multiple testing / FDR + deflated Sharpe -----------------------------------------------

def multiple_testing(prices: pd.DataFrame, grid: list[Params] | None = None, q: float = 0.05) -> dict:
    grid = grid or full_grid()
    sharpes, pvals, per_period = [], [], []
    best = None
    for params in grid:
        d = daily_returns(prices, run(prices, params, HONEST))
        active = d[d != 0]
        s = sharpe(d)
        _, p = stats.ttest_1samp(active, 0.0) if len(active) > 2 else (0.0, 1.0)
        sharpes.append(s); pvals.append(float(p)); per_period.append(active)
        if best is None or s > best[0]:
            best = (s, params, active)
    pvals = np.array(pvals)
    rejected = st.benjamini_hochberg(pvals, q)
    best_sharpe, best_params, best_active = best
    sr_pp = float(best_active.mean() / best_active.std(ddof=1)) if best_active.std(ddof=1) > 0 else 0.0
    per_period_sharpes = np.array([a.mean() / a.std(ddof=1) if len(a) > 2 and a.std(ddof=1) > 0 else 0.0 for a in per_period])
    dsr = st.deflated_sharpe(sr_pp, len(best_active), float(stats.skew(best_active)), float(stats.kurtosis(best_active, fisher=False)), len(grid), float(per_period_sharpes.var(ddof=1)))
    return {
        "variants": len(grid), "best_sharpe": float(best_sharpe), "best_params": best_params,
        "significant_raw": int((pvals < q).sum()), "significant_after_fdr": int(rejected.sum()),
        "expected_max_sharpe_noise": float(st.expected_max_sharpe(len(grid), float(per_period_sharpes.var(ddof=1))) * np.sqrt(252)),
        "deflated_sharpe_prob": dsr,
    }


# --- 8. walk-forward / OOS + confirmatory holdout ----------------------------------------------

def walk_forward(prices: pd.DataFrame, grid: list[Params] | None = None, split: float = 0.6) -> dict:
    grid = grid or full_grid()
    cut = int(len(prices) * split)
    train, hold = prices.iloc[:cut], prices.iloc[cut:]
    best = max(grid, key=lambda p: sharpe(daily_returns(train, run(train, p, HONEST))))
    s_in = sharpe(daily_returns(train, run(train, best, HONEST)))
    s_out = sharpe(daily_returns(hold, run(hold, best, HONEST)))
    return {"best_params": best, "sharpe_in_sample": float(s_in), "sharpe_holdout": float(s_out), "train_bars": cut, "holdout_bars": len(hold)}


# --- 9. matched random-entry control -----------------------------------------------------------

def random_control(prices: pd.DataFrame, params: Params, n_controls: int = 300, seed: int = 0) -> dict:
    """Random entries with the strategy's trade count, holding-period distribution and stop width."""
    strategy = run(prices, params, HONEST)
    s_strat = sharpe(daily_returns(prices, strategy))
    holds = np.array([t.hold for t in strategy])
    n_trades = len(strategy)
    a = atr(prices)
    valid = np.nonzero(~np.isnan(a))[0]
    rng = np.random.default_rng(seed)
    control_sharpes = []
    for _ in range(n_controls):
        entries = np.zeros(len(prices), dtype=bool)
        hold_at = np.zeros(len(prices), dtype=int)
        picks = rng.choice(valid, size=min(n_trades * 3, len(valid)), replace=False)
        picks.sort()
        sampled_holds = rng.choice(holds, size=len(picks), replace=True)
        entries[picks] = True
        hold_at[picks] = sampled_holds
        trades = _walk(prices, entries, params, HONEST, a, holds=hold_at)[:n_trades]
        control_sharpes.append(sharpe(daily_returns(prices, trades)))
    control_sharpes = np.array(control_sharpes)
    return {
        "strategy_sharpe": float(s_strat), "controls": n_controls,
        "control_mean": float(control_sharpes.mean()), "control_sd": float(control_sharpes.std(ddof=1)),
        "percentile": float((control_sharpes < s_strat).mean() * 100.0),
        "matched_trades": n_trades, "median_hold": float(np.median(holds)) if len(holds) else 0.0,
    }
