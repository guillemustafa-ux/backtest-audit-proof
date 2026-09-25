"""Price data: a synthetic random walk with a known truth (no edge), and a CSV loader for the appendix."""

from __future__ import annotations

import numpy as np
import pandas as pd

COLUMNS = ["open", "high", "low", "close"]


def synthetic_prices(n_bars: int = 3000, seed: int = 7, daily_vol: float = 0.012, start: float = 100.0) -> pd.DataFrame:
    """Daily OHLC bars from a driftless log random walk.

    Close-to-close returns are i.i.d. normal with zero mean, so no rule based on past prices has
    an expected edge. Intrabar wicks are fat-tailed (Student t, 3 degrees of freedom) like real
    range days, so stops and targets can both be hit inside one bar: that is what the same-bar
    artifact needs. Tail shape does not create an edge; only the mean does, and it is zero.
    """
    rng = np.random.default_rng(seed)
    log_ret = rng.normal(0.0, daily_vol, n_bars)
    close = start * np.exp(np.cumsum(log_ret))
    open_ = np.empty(n_bars)
    open_[0] = start
    open_[1:] = close[:-1] * np.exp(rng.normal(0.0, daily_vol * 0.25, n_bars - 1))
    body_hi = np.maximum(open_, close)
    body_lo = np.minimum(open_, close)
    wick = np.abs(rng.standard_t(3, (2, n_bars))) * daily_vol * 0.5 * close
    high = body_hi + wick[0]
    low = body_lo - wick[1]
    index = pd.bdate_range("2014-01-01", periods=n_bars)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=index)


def load_csv(path: str) -> pd.DataFrame:
    """Load a daily OHLC CSV with a date column; column names are lower-cased."""
    frame = pd.read_csv(path)
    frame.columns = [c.strip().lower() for c in frame.columns]
    date_col = "date" if "date" in frame.columns else frame.columns[0]
    frame[date_col] = pd.to_datetime(frame[date_col])
    frame = frame.set_index(date_col).sort_index()
    missing = [c for c in COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")
    return frame[COLUMNS].astype(float)
