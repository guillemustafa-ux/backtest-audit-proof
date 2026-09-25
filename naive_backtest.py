"""The backtest that looks great. Every number it prints is an artifact: the data has no edge.

Run: python naive_backtest.py
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from audit.data import synthetic_prices
from audit.engine import Params, Rules, daily_returns, run, summary

NAIVE_RULES = Rules(lookahead=True, same_bar_best_case=True, cost_bps=0.0)
GRID = [Params(lookback=lb, target_atr=t, stop_atr=s) for lb in (5, 10, 20, 40, 60) for t in (1.5, 2.0, 3.0, 4.0) for s in (1.0, 1.5, 2.0, 3.0)]


def best_of_grid(prices, rules=NAIVE_RULES):
    """Pick the parameter set with the highest Sharpe over the whole sample (no holdout, no FDR)."""
    results = []
    for params in GRID:
        trades = run(prices, params, rules)
        s = summary(prices, trades, rules.cost_bps)
        results.append((s["sharpe"], params, trades, s))
    results.sort(key=lambda r: r[0], reverse=True)
    return results


def one_sided_pvalue(prices, trades, cost_bps=0.0):
    """Test 'mean daily return > 0' after already seeing that it is positive."""
    d = daily_returns(prices, trades, cost_bps)
    d = d[d != 0]
    t, p_two = stats.ttest_1samp(d, 0.0)
    return float(t), float(p_two / 2)


if __name__ == "__main__":
    prices = synthetic_prices()
    results = best_of_grid(prices)
    sharpe_, params, trades, s = results[0]
    t, p = one_sided_pvalue(prices, trades)
    print(f"bars: {len(prices)}  variants tried: {len(GRID)}")
    print(f"best params: lookback={params.lookback} target={params.target_atr} ATR stop={params.stop_atr} ATR")
    print(f"trades: {s['trades']}  win rate: {s['win_rate']:.1%}  total return: {s['total_return']:.1%}")
    print(f"Sharpe: {sharpe_:.2f}  t-stat: {t:.2f}  one-sided p: {p:.2e}")
    print("Verdict of the naive backtest: a robust, statistically significant edge.")
    print("Truth: the data is a driftless random walk. Run `python -m audit` to see where the edge comes from.")
