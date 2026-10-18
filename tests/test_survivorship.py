"""Survivorship: a universe built as of today keeps only the symbols that never failed, and that selection alone creates drift."""

import numpy as np
import pandas as pd
import pytest

from audit import report, steps
from audit.data import COLUMNS, synthetic_prices, synthetic_universe
from audit.engine import Params, run
from naive_backtest import GRID

PARAMS = Params(lookback=5, target_atr=0.75, stop_atr=1.0)  # the variant the naive grid picks on the synthetic data


def _frame(closes: list[float]) -> pd.DataFrame:
    close = np.array(closes, dtype=float)
    return pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99, "close": close}, index=pd.bdate_range("2020-01-01", periods=len(close)))


def test_synthetic_universe_is_deterministic_and_engine_shaped():
    a, b = synthetic_universe(3, 200, 7), synthetic_universe(3, 200, 7)
    assert len(a) == 3 and all(len(f) == 200 and list(f.columns) == COLUMNS for f in a)
    assert all(x.equals(y) for x, y in zip(a, b))
    assert not a[0].equals(a[1]) and not a[0].equals(synthetic_prices(200, seed=7))  # distinct from each other and from the single series


def test_delisting_bar_is_the_first_close_below_the_floor():
    assert steps._delisting_bar(_frame([100.0] * 30), 0.5) is None
    assert steps._delisting_bar(_frame([100.0] * 10 + [40.0] + [100.0] * 19), 0.5) == 10
    assert steps._delisting_bar(_frame([100.0, 50.0, 49.9]), 0.5) == 2  # strictly below the floor


def test_point_in_time_symbol_stops_at_its_delisting_bar():
    survivor = _frame([100.0] * 30)
    failed = _frame([100.0] * 10 + [40.0] + [100.0] * 19)  # recovers after delisting; the point-in-time universe never sees that
    r = steps.survivorship([survivor, failed], PARAMS)
    assert (r["n_symbols"], r["delisted"], r["survivors"], r["floor"]) == (2, 1, 1, 0.5)
    assert np.isclose(r["survivors_mean_price_return"], 0.0) and np.isclose(r["all_mean_price_return"], -0.3)  # (0 + (40 / 100 - 1)) / 2
    with pytest.raises(ValueError, match="every symbol was delisted"):
        steps.survivorship([failed, failed], PARAMS)


def test_open_position_is_closed_at_the_delisting_close():
    sym = _frame([100.0] * 20 + [110.0] + [100.0, 80.0, 60.0, 40.0] + [100.0] * 10)  # breakout on bar 20, entry on bar 21, delisted on bar 24
    wide = Params(lookback=5, target_atr=100.0, stop_atr=100.0)  # neither target nor stop is reachable: only time or the delisting closes the trade
    bar = steps._delisting_bar(sym, 0.5)
    assert bar == 24
    last = run(steps._history(sym, bar), wide, steps.HONEST)[-1]
    assert (last.entry_idx, last.exit_idx, last.reason, last.exit_price) == (21, bar, "time", 40.0)
    assert run(sym, wide, steps.HONEST)[-1].exit_idx == 31  # on the full history the same trade runs its 10 bars


def test_survivors_only_universe_shows_drift_the_point_in_time_universe_lacks():
    r = steps.survivorship(synthetic_universe(40, 3000, 7), PARAMS)
    assert r["delisted"] > 0 and r["survivors"] + r["delisted"] == 40
    assert r["survivors_mean_price_return"] > r["all_mean_price_return"]
    assert r["survivors_mean_symbol_sharpe"] > r["point_in_time_mean_symbol_sharpe"]
    assert r["survivors_sharpe"] > r["point_in_time_sharpe"]
    assert r["point_in_time_trades"] >= r["survivors_trades"] > 0


def test_report_renders_row_11_only_with_a_universe():
    prices = synthetic_prices(500, seed=11)
    with_universe = report.render(report.run_audit(prices, GRID, n_controls=10, universe=synthetic_universe(4, 500, 11)), "test")
    assert "| 11 |" in with_universe and "mean per-symbol Sharpe" in with_universe and "Build the universe point-in-time" in with_universe
    assert "Rows 4-10" in with_universe and "Row 11 corrects the data" in with_universe
    without = report.render(report.run_audit(prices, GRID, n_controls=10), "test")
    assert "| 11 |" not in without and "Rows 4-10" in without and "Build the universe point-in-time" not in without
    assert "Row 11" not in without
