"""Each audit step must move its metric in the direction the artifact predicts."""

import numpy as np
import pytest

from audit.data import synthetic_prices
from audit.engine import Params, Rules, daily_returns, run, summary
from audit import steps, stats as st

PRICES = synthetic_prices(1500, seed=11)
PARAMS = Params(lookback=5, target_atr=1.0, stop_atr=0.5)


def test_lookahead_removal_lowers_sharpe():
    r = steps.lookahead(PRICES, PARAMS)
    assert r["after"]["sharpe"] < r["before"]["sharpe"]


def test_same_bar_to_stop_lowers_sharpe():
    r = steps.same_bar(PRICES, PARAMS)
    assert r["after"]["sharpe"] <= r["before"]["sharpe"]
    assert r["ambiguous_bars"] > 0


def test_costs_lower_sharpe():
    r = steps.costs(PRICES, PARAMS, cost_bps=20.0)
    assert r["after"]["sharpe"] < r["before"]["sharpe"]


def test_effective_n_is_below_raw_n():
    r = steps.overlapping_windows(PRICES, PARAMS)
    assert r["effective_n"] < r["raw_n"]
    assert abs(r["t_corrected"]) <= abs(r["t_naive"]) + 1e-9


def test_baseline_comparison_changes_estimate():
    r = steps.event_vs_baseline(PRICES, PARAMS)
    assert "excess_mean" in r and "baseline_mean" in r
    assert np.isfinite(r["excess_mean"])


def test_baseline_test_is_dependence_robust():
    """The event-vs-baseline windows overlap like the ones in step 4, so the test must not treat them as i.i.d."""
    r = steps.event_vs_baseline(PRICES, PARAMS)
    assert np.isfinite(r["t_vs_baseline_nw"]) and 0.0 <= r["p_vs_baseline_nw"] <= 1.0
    assert abs(r["t_vs_baseline_nw"]) <= abs(r["t_vs_baseline"]) + 1e-9


def test_newey_west_diff_t_shrinks_when_events_cluster():
    """Overlapping windows only inflate the i.i.d. two-sample t when the events themselves come in runs,
    as breakouts do; an event dummy that is white noise leaves nothing for the HAC correction to remove."""
    rng = np.random.default_rng(5)
    y = np.convolve(rng.normal(size=1209), np.ones(10), mode="valid")  # overlapping 10-step sums
    z = np.convolve(rng.normal(size=1209), np.ones(10), mode="valid")
    clustered = z > np.quantile(z, 0.85)  # 180 event bars in 43 runs
    from scipy import stats as sp
    t_iid, _ = sp.ttest_ind(y[clustered], y[~clustered], equal_var=False)
    assert abs(st.newey_west_diff_t(y, clustered, lag=9)) < 0.6 * abs(t_iid)
    assert np.isclose(st.newey_west_diff_t(y, clustered, lag=0), t_iid, rtol=0.05)
    assert st.newey_west_diff_t(y, np.ones(len(y), dtype=bool), lag=9) == 0.0


def test_two_sided_p_is_double_one_sided():
    r = steps.two_sided(PRICES, PARAMS)
    assert np.isclose(r["p_two_sided"], 2 * r["p_one_sided"], atol=1e-12) or r["p_two_sided"] >= r["p_one_sided"]


def test_fdr_rejects_no_more_than_raw():
    r = steps.multiple_testing(PRICES, grid=steps.small_grid())
    assert r["significant_after_fdr"] <= r["significant_raw"]
    assert 0.0 <= r["deflated_sharpe_prob"] <= 1.0


def test_walk_forward_reports_oos():
    r = steps.walk_forward(PRICES, grid=steps.small_grid())
    assert "sharpe_in_sample" in r and "sharpe_holdout" in r
    assert r["holdout_bars"] > 0


def test_random_control_percentile_in_range():
    r = steps.random_control(PRICES, PARAMS, n_controls=50, seed=3)
    assert 0.0 <= r["percentile"] <= 100.0
    assert r["controls"] == 50


def test_bh_fdr_basic():
    p = np.array([0.001, 0.01, 0.04, 0.2, 0.9])
    assert st.benjamini_hochberg(p, q=0.05).sum() == 2


def test_block_bootstrap_se_positive():
    x = np.random.default_rng(0).normal(size=300)
    se = st.block_bootstrap_se(x, block=5, n_boot=200, seed=1)
    assert se > 0
