import numpy as np

from audit.data import synthetic_prices
from audit.engine import Params, Rules, daily_returns, run, sharpe, summary


def test_synthetic_prices_are_consistent_ohlc():
    p = synthetic_prices(500, seed=1)
    assert (p["high"] >= p[["open", "close"]].max(axis=1)).all()
    assert (p["low"] <= p[["open", "close"]].min(axis=1)).all()
    assert abs(np.log(p["close"]).diff().dropna().mean()) < 0.002


def test_positions_do_not_overlap():
    p = synthetic_prices(1000, seed=2)
    trades = run(p, Params(), Rules())
    for a, b in zip(trades, trades[1:]):
        assert b.entry_idx > a.exit_idx


def test_entry_is_after_signal_without_lookahead():
    p = synthetic_prices(1000, seed=3)
    from audit.engine import breakout_signals
    sig = breakout_signals(p, 20)
    for t in run(p, Params(lookback=20), Rules(lookahead=False)):
        assert sig[t.entry_idx - 1]


def test_daily_returns_compound_to_trade_returns():
    p = synthetic_prices(600, seed=4)
    trades = run(p, Params(), Rules())
    daily = daily_returns(p, trades)
    for t in trades:
        seg = daily.iloc[t.entry_idx : t.exit_idx + 1]
        assert np.isclose(np.prod(1 + seg) - 1, t.ret, atol=1e-9)


def test_sharpe_of_flat_series_is_zero():
    import pandas as pd
    assert sharpe(pd.Series(np.zeros(100))) == 0.0


def test_summary_keys():
    p = synthetic_prices(400, seed=5)
    s = summary(p, run(p))
    assert set(s) == {"trades", "win_rate", "mean_trade_ret", "sharpe", "total_return"}
