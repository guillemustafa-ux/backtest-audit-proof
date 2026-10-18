"""Turn the audit steps into the written review a methodology audit is expected to deliver."""

from __future__ import annotations

import pandas as pd

from audit import steps
from audit.engine import Params, run, summary


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _f(x: float, d: int = 2) -> str:
    return f"{x:.{d}f}"


def run_audit(prices: pd.DataFrame, grid: list[Params], n_controls: int = 300, cost_bps: float = 20.0) -> dict:
    """Run the naive backtest, then every audit step on the parameters the naive grid picked."""
    from naive_backtest import best_of_grid, one_sided_pvalue

    ranked = best_of_grid(prices)
    naive_sharpe, params, naive_trades, naive_summary = ranked[0]
    t_naive, p_naive = one_sided_pvalue(prices, naive_trades)
    return {
        "bars": len(prices),
        "params": params,
        "naive": {**naive_summary, "t": t_naive, "p_one_sided": p_naive, "variants": len(grid)},
        "lookahead": steps.lookahead(prices, params),
        "same_bar": steps.same_bar(prices, params),
        "costs": steps.costs(prices, params, cost_bps),
        "windows": steps.overlapping_windows(prices, params),
        "baseline": steps.event_vs_baseline(prices, params),
        "two_sided": steps.two_sided(prices, params),
        "multiple": steps.multiple_testing(prices, grid),
        "walk_forward": steps.walk_forward(prices, grid),
        "control": steps.random_control(prices, params, n_controls=n_controls),
        "sensitivity": steps.parameter_sensitivity(prices, grid, params),
        "risk": steps.risk_profile(prices, params, cost_bps),
    }


def render(r: dict, source: str, synthetic: bool = True) -> str:
    p = r["params"]
    nv, la, sb, co, wi, ba, ts, mt, wf, rc, se, rk = (r[k] for k in ("naive", "lookahead", "same_bar", "costs", "windows", "baseline", "two_sided", "multiple", "walk_forward", "control", "sensitivity", "risk"))
    honest_sharpe = co["after"]["sharpe"]
    cl, cr = rk["claim"], rk["corrected"]
    lines = [
        "# Backtest methodology audit",
        "",
        f"Data: {source}, {r['bars']} daily bars. Strategy under review: long breakout above the {p.lookback}-bar high, "
        f"target {p.target_atr} ATR, stop {p.stop_atr} ATR, time exit after {p.max_hold} bars. "
        f"These parameters are the best of {nv['variants']} variants tried by the original backtest.",
        "",
        "## The claim under review",
        "",
        f"The submitted backtest reports {nv['trades']} trades, a {_pct(nv['win_rate'])} win rate, a total return of {_pct(nv['total_return'])}, "
        f"an annualized Sharpe ratio of {_f(nv['sharpe'])} and a one-sided p-value of {nv['p_one_sided']:.1e} (t = {_f(nv['t'])}). "
        "The audit below removes one methodological artifact at a time and reports what is left.",
        "",
        "## Findings, artifact by artifact",
        "",
        "| # | Artifact | Metric before | Metric after | Severity |",
        "|---|---|---|---|---|",
        f"| 1 | Look-ahead: the signal uses the close of the bar it enters on | Sharpe {_f(la['before']['sharpe'])} | Sharpe {_f(la['after']['sharpe'])} | {_sev(la['before']['sharpe'] - la['after']['sharpe'])} |",
        f"| 2 | Same-bar ambiguity resolved in favour of the target ({sb['ambiguous_bars']} of {sb['trades']} exits) | Sharpe {_f(sb['before']['sharpe'])} | Sharpe {_f(sb['after']['sharpe'])} | {_sev(sb['before']['sharpe'] - sb['after']['sharpe'])} |",
        f"| 3 | Zero transaction costs (audit applies {co['cost_bps']:.0f} bps round trip) | Sharpe {_f(co['before']['sharpe'])} | Sharpe {_f(co['after']['sharpe'])} | {_sev(co['before']['sharpe'] - co['after']['sharpe'])} |",
        f"| 4 | Overlapping {wi['horizon']}-bar forward-return windows counted as independent | t = {_f(wi['t_naive'])} on N = {wi['raw_n']} | t = {_f(wi['t_corrected'])} (Newey-West), {_f(wi['t_bootstrap'])} (block bootstrap), effective N = {wi['effective_n']:.0f} | {_sev_t(wi['t_naive'], wi['t_corrected'])} |",
        f"| 5 | Event returns tested against zero instead of the baseline | t vs 0 = {_f(ba['t_vs_zero'])} | t vs baseline = {_f(ba['t_vs_baseline'])} (i.i.d.), {_f(ba['t_vs_baseline_nw'])} (Newey-West, p = {ba['p_vs_baseline_nw']:.2f}), excess {_pct(ba['excess_mean'])} per window | {_sev_t(ba['t_vs_zero'], ba['t_vs_baseline_nw'])} |",
        f"| 6 | One-sided test chosen after seeing the sign | p = {ts['p_one_sided']:.3f} | p = {ts['p_two_sided']:.3f} (two-sided) | {'material' if ts['p_one_sided'] < 0.05 <= ts['p_two_sided'] else 'minor'} |",
        f"| 7 | Best of {mt['variants']} variants reported without multiple-testing control | {mt['significant_raw']} variants significant at 5% | {mt['significant_after_fdr']} after Benjamini-Hochberg; expected best Sharpe from noise alone {_f(mt['expected_max_sharpe_noise'])}; deflated-Sharpe probability {_f(mt['deflated_sharpe_prob'])} | {'material' if mt['deflated_sharpe_prob'] < 0.95 else 'minor'} |",
        f"| 8 | Parameters optimized on the full sample, no holdout | in-sample Sharpe {_f(wf['sharpe_in_sample'])} ({wf['train_bars']} bars) | holdout Sharpe {_f(wf['sharpe_holdout'])} ({wf['holdout_bars']} bars, touched once) | {_sev(wf['sharpe_in_sample'] - wf['sharpe_holdout'])} |",
        f"| 9 | No random-entry control | strategy Sharpe {_f(rc['strategy_sharpe'])} | {rc['controls']} matched random-entry runs: mean {_f(rc['control_mean'])}, sd {_f(rc['control_sd'])}; strategy at the {rc['percentile']:.0f}th percentile | {'material' if rc['percentile'] < 95 else 'minor'} |",
        f"| 10 | Only the best parameter point is reported; the surface around it is hidden | reported variant Sharpe {_f(se['reported_sharpe'])} (rank {se['rank']} of {len(se['surface'])}) | its {se['neighbours']} one-step neighbours: mean {_f(se['neighbour_mean'])} (min {_f(se['neighbour_min'])}, max {_f(se['neighbour_max'])}); grid median {_f(se['grid_median'])}; {_pct(se['share_within_0_3'])} of neighbours within 0.3 Sharpe | {_sev(se['reported_sharpe'] - se['neighbour_mean'])} |",
        "",
        "Rows 1-3 are corrections to the backtest itself and compound: after all three the honest Sharpe is "
        f"{_f(honest_sharpe)}. Rows 4-10 are corrections to the evidence: they do not change the returns, they change how much the returns prove.",
        "",
        "## Risk and exposure, claim vs corrected",
        "",
        "| Metric | Claim (as submitted) | Corrected (rows 1-3 applied) |",
        "|---|---|---|",
        f"| Max drawdown | {_pct(cl['max_drawdown'])} over {cl['drawdown_bars']} bars | {_pct(cr['max_drawdown'])} over {cr['drawdown_bars']} bars |",
        f"| Annualised volatility | {_pct(cl['ann_vol'])} | {_pct(cr['ann_vol'])} |",
        f"| Exposure (share of bars in a position) | {_pct(cl['exposure'])} | {_pct(cr['exposure'])} |",
        f"| Turnover (trades per year) | {_f(cl['trades_per_year'], 1)} | {_f(cr['trades_per_year'], 1)} |",
        f"| Worst day | {_pct(cl['worst_day'])} | {_pct(cr['worst_day'])} |",
        f"| Expected shortfall, worst 5% of days | {_pct(cl['es_5'])} | {_pct(cr['es_5'])} |",
        "",
        f"The corrected column is the same strategy with look-ahead removed, ambiguous bars booked to the stop and {co['cost_bps']:.0f} bps charged per round trip; "
        "it is what the risk figures would have been had the backtest been built correctly.",
        "",
        "## Material weaknesses",
        "",
    ]
    lines += [f"- {w}" for w in _weaknesses(r)]
    lines += [
        "",
        "## Required corrections",
        "",
        "- Compute every signal from information available at the decision time and enter at the next bar's open, or later; re-run the full grid under that rule before any other analysis.",
        "- When a bar touches both target and stop, book the stop. If the order of touches matters to the strategy, the test needs intrabar data; daily bars cannot resolve it.",
        "- Charge a documented round-trip cost (spread, slippage, commission) on every trade and report the cost assumption next to every performance figure.",
        "- Report the effective sample size and a dependence-robust t-statistic (Newey-West or block bootstrap with blocks at least as long as the holding period) instead of the i.i.d. t-test.",
        "- Test event returns against the unconditional baseline over the same horizon, not against zero, with the same dependence-robust standard errors: the baseline windows overlap too.",
        "- Pre-register the hypothesis direction; use two-sided tests unless the direction was fixed before looking at the data.",
        "- Control the family-wise or false-discovery rate across every variant tried, and report the deflated Sharpe ratio with the number of trials.",
        "- Keep a confirmatory holdout that is touched exactly once, after the methodology is frozen.",
        "- Compare against a matched random-entry control: same number of trades, same holding-period distribution, same stop width, per instrument.",
        "- Report the parameter surface, not its maximum: the one-step neighbourhood and the grid distribution next to the chosen variant.",
        "- Report maximum drawdown, exposure and turnover with every performance figure; a Sharpe ratio without them hides how the return was earned.",
        "",
        "## Recommended methodology for the next pre-registered run",
        "",
        "1. Freeze the rule set, the parameter grid, the cost model and the holdout boundary in a written pre-registration before running anything.",
        "2. Split the history into a development sample and a confirmatory holdout; do all selection on the development sample only.",
        "3. For every variant, record the Sharpe ratio, a dependence-robust t-statistic, the effective N and the number of ambiguous bars.",
        "4. Select with Benjamini-Hochberg at q = 0.05 and report the deflated Sharpe ratio of the chosen variant against the full trial count.",
        "5. Run the matched random-entry control on the development sample; a variant that does not beat the 95th percentile of the control does not proceed.",
        "6. Run the surviving variant once on the holdout. Report it whatever the result; a second look at the holdout invalidates it.",
        "7. Report the parameter neighbourhood of the surviving variant and its risk profile (drawdown, exposure, turnover) with the holdout result.",
        "",
        "## Evidence required before confirmatory testing",
        "",
        "- A signed pre-registration with the frozen rules, grid, cost model and holdout boundary.",
        "- The development-sample table for every variant, with raw and FDR-adjusted p-values and the deflated Sharpe ratio.",
        "- The random-entry control distribution and the chosen variant's percentile within it.",
        "- An intrabar or tick-level check of same-bar exits if they exceed 1% of trades.",
        "- A reproducibility run: the same numbers from a clean checkout with the recorded seed.",
        "",
        "---",
        "",
        "Generated by `python -m audit`. " + (
            "The synthetic data has no edge by construction, so every unit of performance above zero in the claim is an artifact; the audit quantifies which artifact contributes how much."
            if synthetic else
            "On real data the truth is unknown: the audit cannot say the edge is zero, only how much of the claim survives each correction and how much the baseline drift explains."
        ),
        "",
    ]
    return "\n".join(lines)


def _sev(delta: float) -> str:
    if delta >= 1.0:
        return "critical"
    if delta >= 0.3:
        return "material"
    return "minor"


def _sev_t(t_before: float, t_after: float) -> str:
    """Severity of an evidence artifact: it matters when it manufactures significance."""
    if abs(t_before) >= 1.96 > abs(t_after):
        return "material: significance disappears"
    if abs(t_before) >= 1.96 and abs(t_before) - abs(t_after) >= 1.0:
        return "material"
    return "minor"


def _weaknesses(r: dict) -> list[str]:
    la, sb, co, wi, mt, wf, rc, se, rk = (r[k] for k in ("lookahead", "same_bar", "costs", "windows", "multiple", "walk_forward", "control", "sensitivity", "risk"))
    out = []
    d = la["before"]["sharpe"] - la["after"]["sharpe"]
    if d >= 0.3:
        out.append(f"Look-ahead is the largest single artifact: {_f(d)} Sharpe points come from knowing the close before entering.")
    d = sb["before"]["sharpe"] - sb["after"]["sharpe"]
    if sb["ambiguous_bars"] > 0:
        out.append(f"{sb['ambiguous_bars']} exits are ambiguous within the bar; resolving them in the strategy's favour is worth {_f(d)} Sharpe points.")
    d = co["before"]["sharpe"] - co["after"]["sharpe"]
    out.append(f"With {co['cost_bps']:.0f} bps of round-trip cost the Sharpe falls by {_f(d)}; the claim assumes costless execution.")
    if wi["effective_n"] < wi["raw_n"] * 0.7:
        out.append(f"The {wi['raw_n']} signal windows are not independent: effective N is about {wi['effective_n']:.0f}, so the naive t-statistic overstates the evidence.")
    if mt["significant_after_fdr"] < mt["significant_raw"]:
        out.append(f"Of {mt['variants']} variants, {mt['significant_raw']} look significant at 5% (noise alone would produce about {mt['variants'] * 0.05:.0f}) and {mt['significant_after_fdr']} survive FDR control; the best Sharpe of pure noise across that many trials would be about {_f(mt['expected_max_sharpe_noise'])}.")
    if wf["sharpe_holdout"] < wf["sharpe_in_sample"]:
        out.append(f"The in-sample Sharpe of {_f(wf['sharpe_in_sample'])} becomes {_f(wf['sharpe_holdout'])} on a holdout that was never used for selection.")
    if rc["percentile"] < 95:
        out.append(f"Random entries with the same trade count, holding periods and stop width reach the strategy's Sharpe in {100 - rc['percentile']:.0f}% of runs; the entry rule adds nothing measurable.")
    d = se["reported_sharpe"] - se["neighbour_mean"]
    if d >= 0.3:
        out.append(f"The reported parameters sit on a spike: their one-step neighbours average a Sharpe of {_f(se['neighbour_mean'])} against the reported {_f(se['reported_sharpe'])}, so the result depends on one point of the grid.")
    if rk["corrected"]["max_drawdown"] > rk["claim"]["max_drawdown"]:
        out.append(f"The corrected run's maximum drawdown is {_pct(rk['corrected']['max_drawdown'])} against {_pct(rk['claim']['max_drawdown'])} in the claim.")
    return out or ["No material weakness found by the ten checks."]
