"""The audit steps. Each takes the naive setup, removes or exposes one artifact, and returns the numbers for the report."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
from scipy import stats

from audit import stats as st
from audit.engine import TRADING_DAYS, Params, Rules, Trade, _walk, atr, breakout_signals, daily_returns, run, sharpe, summary

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
    # The windows overlap exactly as in step 4, so the two-sample test above is only the i.i.d. reference point.
    t_nw = st.newey_west_diff_t(base, signals[valid], lag=horizon - 1)
    p_nw = float(2.0 * stats.norm.sf(abs(t_nw)))
    return {
        "event_mean": float(event.mean()), "baseline_mean": float(base.mean()), "excess_mean": float(event.mean() - base.mean()),
        "t_vs_zero": float(t_vs_zero), "t_vs_baseline": float(t_vs_base), "p_vs_baseline": float(p_vs_base),
        "t_vs_baseline_nw": t_nw, "p_vs_baseline_nw": p_nw,
    }


# --- 6. directional vs two-sided ---------------------------------------------------------------

def two_sided(prices: pd.DataFrame, params: Params) -> dict:
    """A direction chosen after seeing the sign always halves the p-value, whichever sign it is."""
    d = daily_returns(prices, run(prices, params, HONEST))
    d = d[d != 0]
    t, p_two = stats.ttest_1samp(d, 0.0)
    return {"t": float(t), "p_one_sided": float(p_two / 2), "p_two_sided": float(p_two), "n": int(len(d))}


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
    control_trades = []
    for _ in range(n_controls):
        # Entry dates spread over the whole sample, like the strategy's; overlaps are dropped by the walk.
        entries = np.zeros(len(prices), dtype=bool)
        hold_at = np.zeros(len(prices), dtype=int)
        picks = np.sort(rng.choice(valid, size=min(n_trades, len(valid)), replace=False))
        entries[picks] = True
        hold_at[picks] = rng.choice(holds, size=len(picks), replace=True)
        trades = _walk(prices, entries, params, HONEST, a, holds=hold_at)
        control_trades.append(len(trades))
        control_sharpes.append(sharpe(daily_returns(prices, trades)))
    control_sharpes = np.array(control_sharpes)
    return {
        "strategy_sharpe": float(s_strat), "controls": n_controls,
        "control_mean": float(control_sharpes.mean()), "control_sd": float(control_sharpes.std(ddof=1)),
        "percentile": float((control_sharpes < s_strat).mean() * 100.0),
        "matched_trades": n_trades, "control_trades_mean": float(np.mean(control_trades)),
        "median_hold": float(np.median(holds)) if len(holds) else 0.0,
    }


# --- 10. parameter sensitivity -----------------------------------------------------------------

def _key(p: Params) -> tuple:
    return (p.lookback, p.target_atr, p.stop_atr)


def _neighbours(params: Params, grid: list[Params]) -> list[Params]:
    """Variants that differ from `params` in exactly one parameter, by exactly one step of its sorted axis."""
    out = []
    for name in ("lookback", "target_atr", "stop_atr"):
        axis = sorted({getattr(p, name) for p in grid})
        at = axis.index(getattr(params, name))
        out += [replace(params, **{name: axis[i]}) for i in (at - 1, at + 1) if 0 <= i < len(axis)]
    return out


def parameter_sensitivity(prices: pd.DataFrame, grid: list[Params], params: Params, rules: Rules = NAIVE) -> dict:
    """The reported number is the peak of a search; the one-step neighbours say how typical that peak is.

    Run under the NAIVE rules the surface is the one the claim was selected on, so the row audits the
    claim itself. An interior point of the 5x5x5 grid has 6 neighbours, a corner has 3.
    """
    keys = {_key(p) for p in grid}
    if _key(params) not in keys:
        raise ValueError(f"params not in grid: {_key(params)}")
    near_params = [p for p in _neighbours(params, grid) if _key(p) in keys]
    if not near_params:
        raise ValueError("params has no grid neighbours: the grid needs a second value on at least one axis")
    surface = {_key(p): sharpe(daily_returns(prices, run(prices, p, rules), rules.cost_bps)) for p in grid}
    reported = surface[_key(params)]
    near = np.array([surface[_key(p)] for p in near_params])
    values = np.array(list(surface.values()))
    return {
        "rules": "naive" if rules == NAIVE else "honest" if rules == HONEST else "custom", "reported_sharpe": float(reported),
        "neighbour_params": [_key(p) for p in near_params], "neighbour_sharpes": [float(s) for s in near],
        "neighbour_mean": float(near.mean()), "neighbour_min": float(near.min()), "neighbour_max": float(near.max()),
        "neighbours": int(len(near)), "grid_median": float(np.median(values)), "grid_mean": float(values.mean()),
        "within_0_3": int((np.abs(near - reported) <= 0.3).sum()),
        "surface": surface, "rank": int((values > reported).sum()) + 1,
    }


# --- 11. survivorship bias ---------------------------------------------------------------------

def _delisting_bar(prices: pd.DataFrame, floor: float) -> int | None:
    """Index of the first close below `floor` times the first close, or None when the symbol never gets there."""
    close = prices["close"].to_numpy()
    below = np.nonzero(close < floor * close[0])[0]
    return int(below[0]) if len(below) else None


def _history(prices: pd.DataFrame, delisting_bar: int | None) -> pd.DataFrame:
    """The symbol's data as the point-in-time universe sees it: all of it for a survivor, up to and including the delisting bar otherwise."""
    return prices if delisting_bar is None else prices.iloc[: delisting_bar + 1]


def _equal_weight(series: list[pd.Series], bars: int) -> pd.Series:
    """Per bar, the mean daily return of the symbols that still exist; a delisted symbol leaves the denominator too."""
    table = np.full((len(series), bars), np.nan)
    for row, s in zip(table, series):
        row[: len(s)] = s.to_numpy()
    alive = ~np.isnan(table)
    return pd.Series(np.where(alive, table, 0.0).sum(axis=0) / np.maximum(alive.sum(axis=0), 1))


def survivorship(universe: list[pd.DataFrame], params: Params, floor: float = 0.5, rules: Rules = HONEST) -> dict:
    """A universe built as of today holds only the symbols that never failed; the point-in-time universe holds
    every symbol until its delisting bar, with the open position closed at that bar's close.

    A symbol is delisted on the first bar whose close is below `floor` times its first close; from that bar on
    its data does not exist. Each symbol runs the strategy on its own history (the whole sample for a survivor,
    the truncated one otherwise; the engine's time exit at the last bar closes whatever is open).

    `*_mean_symbol_sharpe` is the mean of the per-symbol Sharpes, each over the symbol's own history: the number
    the report compares. `*_sharpe` is the equal-weight portfolio of the symbols alive on each bar; it is kept
    for reference but a 23-symbol portfolio and one that fades from 40 to 23 do not diversify alike, so the
    two are not comparable. `*_mean_price_return` is the symbols' buy-and-hold change, not the strategy's return.
    """
    bars = max(len(p) for p in universe)
    delisted_at = [_delisting_bar(p, floor) for p in universe]
    histories = [_history(p, bar) for p, bar in zip(universe, delisted_at)]
    trades = [run(h, params, rules) for h in histories]
    daily = [daily_returns(h, t, rules.cost_bps) for h, t in zip(histories, trades)]
    symbol_sharpes = [sharpe(d) for d in daily]
    price_return = [float(h["close"].iloc[-1] / h["close"].iloc[0] - 1.0) for h in histories]
    survivors = [i for i, bar in enumerate(delisted_at) if bar is None]
    if not survivors:
        raise ValueError("every symbol was delisted: nothing is left to build the as-of-today universe from")
    return {
        "n_symbols": len(universe), "delisted": len(universe) - len(survivors), "survivors": len(survivors), "floor": floor,
        "survivors_sharpe": sharpe(_equal_weight([daily[i] for i in survivors], bars)), "point_in_time_sharpe": sharpe(_equal_weight(daily, bars)),
        "survivors_mean_symbol_sharpe": float(np.mean([symbol_sharpes[i] for i in survivors])), "point_in_time_mean_symbol_sharpe": float(np.mean(symbol_sharpes)),
        "survivors_mean_price_return": float(np.mean([price_return[i] for i in survivors])), "all_mean_price_return": float(np.mean(price_return)),
        "survivors_trades": sum(len(trades[i]) for i in survivors), "point_in_time_trades": sum(len(t) for t in trades),
    }


# --- risk and exposure profile -----------------------------------------------------------------

def _risk(prices: pd.DataFrame, trades: list[Trade], daily: pd.Series) -> dict:
    dd, bars, not_recovered = st.max_drawdown(daily)  # `bars` is the longest time under water, not the length of the deepest fall
    return {
        "max_drawdown": dd, "drawdown_bars": bars, "not_recovered": not_recovered,
        "ann_vol": float(daily.std(ddof=1) * np.sqrt(TRADING_DAYS)),
        "exposure": sum(t.exit_idx - t.entry_idx + 1 for t in trades) / len(prices),
        "trades_per_year": len(trades) / (len(prices) / TRADING_DAYS),
        "worst_day": float(daily.min()), "es_5": st.expected_shortfall(daily, 0.05), "trades": len(trades),
    }


def risk_profile(prices: pd.DataFrame, params: Params, cost_bps: float = 20.0) -> dict:
    """Drawdown, volatility, exposure, turnover and tail of the claim (naive rules, no cost) and of the
    corrected run (rows 1-3 applied: no look-ahead, ambiguous bars to the stop, `cost_bps` per round trip).

    Volatility is the std(ddof=1) of the mark-to-market daily series over every bar, flat days as zeros,
    annualised with sqrt(252): the convention of `engine.sharpe`. Exposure counts the bars from the entry
    bar to the exit bar inclusive, which are the bars `daily_returns` books P&L on. `series` carries both
    daily series as plain arrays for the charts.
    """
    claim = run(prices, params, NAIVE)
    corrected = run(prices, params, Rules(lookahead=False, same_bar_best_case=False, cost_bps=cost_bps))
    d_claim, d_corrected = daily_returns(prices, claim), daily_returns(prices, corrected, cost_bps)
    return {
        "claim": _risk(prices, claim, d_claim),
        "corrected": _risk(prices, corrected, d_corrected),
        "series": {"claim": d_claim.to_numpy(), "corrected": d_corrected.to_numpy()},
    }
