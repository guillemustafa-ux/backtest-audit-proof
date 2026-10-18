"""Risk statistics, the parameter-sensitivity step and the risk profile added to the audit."""

import numpy as np
import pytest

from audit import report, stats as st, steps
from audit.data import synthetic_prices
from audit.engine import Params, Rules
from naive_backtest import GRID, best_of_grid

PRICES = synthetic_prices(1500, seed=11)


def test_max_drawdown_on_hand_made_series():
    dd, bars, not_recovered = st.max_drawdown(np.array([0.1, -0.5, 0.1]))  # equity 1.1, 0.55, 0.605
    assert np.isclose(dd, 0.5) and bars == 2 and not_recovered
    dd, bars, not_recovered = st.max_drawdown(np.array([-0.2, 0.1]))  # the peak is the starting equity of 1.0
    assert np.isclose(dd, 0.2) and bars == 2 and not_recovered
    assert st.max_drawdown(np.array([-0.2, 0.3]))[1:] == (1, False)  # back above the peak on the last bar


def test_max_drawdown_of_flat_or_empty_series_is_zero():
    assert st.max_drawdown(np.zeros(50)) == (0.0, 0, False)
    assert st.max_drawdown(np.array([])) == (0.0, 0, False)


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
    assert len(r["surface"]) == len(GRID)


def test_sensitivity_neighbours_are_one_step_along_each_axis():
    r = steps.parameter_sensitivity(PRICES, GRID, Params(lookback=5, target_atr=0.75, stop_atr=1.0))
    assert set(r["neighbour_params"]) == {(10, 0.75, 1.0), (5, 1.0, 1.0), (5, 0.75, 0.75), (5, 0.75, 1.5)}
    assert r["neighbours"] == len(r["neighbour_sharpes"]) == 4


def test_sensitivity_neighbour_count_depends_on_position_in_grid():
    interior = steps.parameter_sensitivity(PRICES, GRID, Params(lookback=20, target_atr=1.5, stop_atr=1.0))
    corner = steps.parameter_sensitivity(PRICES, GRID, Params(lookback=5, target_atr=0.75, stop_atr=0.5))
    assert interior["neighbours"] == len(interior["neighbour_sharpes"]) == 6
    assert corner["neighbours"] == 3
    assert 0 <= interior["within_0_3"] <= interior["neighbours"]


def test_sensitivity_labels_the_rules():
    point, grid = Params(lookback=5, target_atr=1.0, stop_atr=0.5), steps.small_grid()
    assert steps.parameter_sensitivity(PRICES, grid, point, rules=steps.HONEST)["rules"] == "honest"
    assert steps.parameter_sensitivity(PRICES, grid, point, rules=Rules(cost_bps=5.0))["rules"] == "custom"


def test_sensitivity_rejects_a_grid_without_neighbours():
    point = Params(lookback=5, target_atr=1.0, stop_atr=0.5)
    with pytest.raises(ValueError, match="no grid neighbours"):
        steps.parameter_sensitivity(PRICES, [point], point)


def test_sensitivity_rejects_params_off_the_grid():
    with pytest.raises(ValueError, match="not in grid"):
        steps.parameter_sensitivity(PRICES, steps.small_grid(), Params(lookback=7, target_atr=1.0, stop_atr=0.5))


def test_risk_profile_bounds():
    r = steps.risk_profile(PRICES, Params(lookback=5, target_atr=1.0, stop_atr=0.5), cost_bps=20.0)
    for side in ("claim", "corrected"):
        p = r[side]
        assert p["max_drawdown"] >= 0.0 and p["drawdown_bars"] >= 0 and isinstance(p["not_recovered"], bool)
        assert 0.0 <= p["exposure"] <= 1.0
        assert p["trades_per_year"] > 0 and p["ann_vol"] > 0
        assert p["es_5"] >= p["worst_day"]  # ES averages the worst days, so it is never below the single worst one


def test_corrected_run_draws_down_more_than_the_claim_on_the_synthetic_sample():
    r = steps.risk_profile(PRICES, best_of_grid(PRICES)[0][1], cost_bps=20.0)
    assert r["corrected"]["max_drawdown"] > r["claim"]["max_drawdown"]
    assert r["corrected"]["drawdown_bars"] > r["claim"]["drawdown_bars"]


def test_spike_needs_a_positive_reported_sharpe_and_a_real_gap():
    assert report._spike({"reported_sharpe": 3.86, "neighbour_mean": 1.5})
    assert not report._spike({"reported_sharpe": 3.86, "neighbour_mean": 3.43})  # a plateau
    assert not report._spike({"reported_sharpe": 0.1, "neighbour_mean": 0.04})  # half of almost nothing is not a spike
    assert not report._spike({"reported_sharpe": -0.5, "neighbour_mean": -0.6})
    assert not report._spike({"reported_sharpe": 0.0, "neighbour_mean": -0.01})


def test_under_water_marks_an_unrecovered_run():
    assert report._under_water({"drawdown_bars": 116, "not_recovered": False}) == "116 bars"
    assert report._under_water({"drawdown_bars": 2941, "not_recovered": True}) == "2941 bars (not recovered)"


def test_report_renders_row_10_and_risk_section():
    r = report.run_audit(synthetic_prices(500, seed=11), GRID, n_controls=10)
    text = report.render(r, "test")
    assert "| 10 |" in text and "## Risk and exposure, claim vs corrected" in text
    assert "Rows 4-10" in text and "Row 10 reads the parameter surface" in text
    assert "| Max drawdown |" in text and "| Longest time under water |" in text and "(flat days count as 0)" in text
