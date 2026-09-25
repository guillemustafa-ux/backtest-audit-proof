"""A long-only breakout backtest with switchable artifacts.

The strategy: go long when the close breaks above the highest high of the previous `lookback`
bars; exit at an ATR-multiple target or stop, or at the close after `max_hold` bars.

Every flag in `Rules` is an artifact the naive backtest plants; the audit turns them off one by one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass(frozen=True)
class Params:
    lookback: int = 20
    target_atr: float = 3.0
    stop_atr: float = 1.5
    max_hold: int = 10


@dataclass(frozen=True)
class Rules:
    lookahead: bool = False        # enter at the open of the signal bar (its close is not known yet)
    same_bar_best_case: bool = False  # target and stop both hit in one bar -> count the target
    cost_bps: float = 0.0          # round-trip cost in basis points of the entry price


@dataclass(frozen=True)
class Trade:
    entry_idx: int
    exit_idx: int
    entry_price: float
    exit_price: float
    ret: float
    hold: int
    stop_width: float  # in ATR units
    reason: str


def atr(prices: pd.DataFrame, n: int = 14) -> np.ndarray:
    high, low, close = prices["high"].to_numpy(), prices["low"].to_numpy(), prices["close"].to_numpy()
    prev_close = np.concatenate([[close[0]], close[:-1]])
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    return pd.Series(tr).rolling(n, min_periods=n).mean().to_numpy()


def breakout_signals(prices: pd.DataFrame, lookback: int) -> np.ndarray:
    """True on bars whose close exceeds the highest high of the previous `lookback` bars."""
    high = prices["high"]
    prior_max = high.rolling(lookback, min_periods=lookback).max().shift(1)
    return (prices["close"] > prior_max).to_numpy()


def _walk(prices: pd.DataFrame, entries: np.ndarray, params: Params, rules: Rules, atr_values: np.ndarray, holds: np.ndarray | None = None) -> list[Trade]:
    """Simulate one position at a time from candidate entry bars.

    `entries[i]` is True when a position may open on bar i (already shifted for lookahead).
    `holds`, when given, overrides `params.max_hold` per entry (used by the random control).
    """
    open_, high, low, close = (prices[c].to_numpy() for c in ("open", "high", "low", "close"))
    n = len(prices)
    trades: list[Trade] = []
    i = 0
    while i < n:
        if not entries[i] or np.isnan(atr_values[i]):
            i += 1
            continue
        entry = open_[i]
        a = atr_values[i]
        target = entry + params.target_atr * a
        stop = entry - params.stop_atr * a
        max_hold = int(holds[i]) if holds is not None else params.max_hold
        exit_price, exit_idx, reason = None, None, ""
        j = i
        while j < n and j <= i + max_hold:
            hit_target, hit_stop = high[j] >= target, low[j] <= stop
            if hit_target and hit_stop:
                exit_price, exit_idx, reason = (target, j, "target") if rules.same_bar_best_case else (stop, j, "stop")
                break
            if hit_stop:
                exit_price, exit_idx, reason = stop, j, "stop"
                break
            if hit_target:
                exit_price, exit_idx, reason = target, j, "target"
                break
            j += 1
        if exit_price is None:
            exit_idx = min(i + max_hold, n - 1)
            exit_price, reason = close[exit_idx], "time"
        ret = exit_price / entry - 1.0 - rules.cost_bps / 10_000.0
        trades.append(Trade(i, exit_idx, entry, exit_price, ret, exit_idx - i, params.stop_atr, reason))
        i = exit_idx + 1
    return trades


def run(prices: pd.DataFrame, params: Params = Params(), rules: Rules = Rules()) -> list[Trade]:
    signals = breakout_signals(prices, params.lookback)
    if rules.lookahead:
        entries = signals                      # uses the close of bar i to buy at the open of bar i
    else:
        entries = np.concatenate([[False], signals[:-1]])  # signal on bar i, entry at the open of i+1
    return _walk(prices, entries, params, rules, atr(prices))


def daily_returns(prices: pd.DataFrame, trades: list[Trade], cost_bps: float = 0.0) -> pd.Series:
    """Mark-to-market daily returns of the strategy: 0 when flat, price change while in a position."""
    close = prices["close"].to_numpy()
    out = np.zeros(len(prices))
    for t in trades:
        prev = t.entry_price
        for j in range(t.entry_idx, t.exit_idx + 1):
            price = t.exit_price if j == t.exit_idx else close[j]
            out[j] = price / prev - 1.0
            prev = price
        out[t.exit_idx] -= cost_bps / 10_000.0
    return pd.Series(out, index=prices.index)


def sharpe(daily: pd.Series) -> float:
    sd = daily.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return 0.0
    return float(daily.mean() / sd * np.sqrt(TRADING_DAYS))


def summary(prices: pd.DataFrame, trades: list[Trade], cost_bps: float = 0.0) -> dict:
    daily = daily_returns(prices, trades, cost_bps)
    rets = np.array([t.ret for t in trades])
    return {
        "trades": len(trades),
        "win_rate": float((rets > 0).mean()) if len(rets) else 0.0,
        "mean_trade_ret": float(rets.mean()) if len(rets) else 0.0,
        "sharpe": sharpe(daily),
        "total_return": float(np.prod(1.0 + daily) - 1.0),
    }
