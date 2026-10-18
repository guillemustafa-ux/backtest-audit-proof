"""Risk statistics, the parameter-sensitivity step and the risk profile added to the audit."""

import numpy as np

from audit import report, stats as st, steps
from audit.data import synthetic_prices
from audit.engine import Params
from naive_backtest import GRID, best_of_grid

PRICES = synthetic_prices(1500, seed=11)


def test_max_drawdown_on_hand_made_series():
    dd, bars = st.max_drawdown(np.array([0.1, -0.5, 0.1]))  # equity 1.1, 0.55, 0.605
    assert np.isclose(dd, 0.5) and bars == 2
    dd, bars = st.max_drawdown(np.array([-0.2, 0.1]))  # the peak is the starting equity of 1.0
    assert np.isclose(dd, 0.2) and bars == 2


def test_max_drawdown_of_flat_or_empty_series_is_zero():
    assert st.max_drawdown(np.zeros(50)) == (0.0, 0)
    assert st.max_drawdown(np.array([])) == (0.0, 0)


def test_expected_shortfall_averages_the_worst_share():
    x = np.array([0.01, -0.1, 0.02, 0.0, -0.05, 0.03, 0.01, 0.0, 0.02, 0.01])
    assert np.isclose(st.expected_shortfall(x, alpha=0.2), -0.075)  # ceil(0.2 * 10) = 2 worst days
    assert st.expected_shortfall(np.array([])) == 0.0


def test_sensitivity_reported_variant_is_the_grid_maximum():
    best = best_of_grid(PRICES)[0][1]
    r = steps.parameter_sensitivity(PRICES, GRID, best)
    assert r["rules"] == "naive"
    assert np.isclose(r["reported_sharpe"], max(r["surface"].values()))
    assert r["rank"] == 1
    assert r["neighbour_mean"] <= r["reported_sharpe"]
    assert len(r["surface"]) == len(GRID)


def test_sensitivity_neighbour_count_depends_on_position_in_grid():
    interior = steps.parameter_sensitivity(PRICES, GRID, Params(lookback=20, target_atr=1.5, stop_atr=1.0))
    corner = steps.parameter_sensitivity(PRICES, GRID, Params(lookback=5, target_atr=0.75, stop_atr=0.5))
    assert interior["neighbours"] == len(interior["neighbour_sharpes"]) == 6
    assert corner["neighbours"] == 3
    assert 0.0 <= interior["share_within_0_3"] <= 1.0


def test_risk_profile_bounds():
    r = steps.risk_profile(PRICES, Params(lookback=5, target_atr=1.0, stop_atr=0.5), cost_bps=20.0)
    for side in ("claim", "corrected"):
        p = r[side]
        assert p["max_drawdown"] >= 0.0 and p["drawdown_bars"] >= 0
        assert 0.0 <= p["exposure"] <= 1.0
        assert p["trades_per_year"] > 0 and p["ann_vol"] > 0
        assert p["es_5"] >= p["worst_day"]  # ES averages the worst days, so it is never below the single worst one


def test_report_renders_row_10_and_risk_section():
    r = report.run_audit(synthetic_prices(500, seed=11), GRID, n_controls=10)
    text = report.render(r, "test")
    assert "| 10 |" in text and "## Risk and exposure, claim vs corrected" in text
    assert "Rows 4-10" in text
