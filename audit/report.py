"""Turn the audit steps into the written review a methodology audit is expected to deliver."""

from __future__ import annotations

import numpy as np
import pandas as pd

from audit import steps
from audit.charts import BAND_QUANTILES
from audit.engine import Params, run, summary


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _f(x: float, d: int = 2) -> str:
    return f"{x:.{d}f}"


def _ordinal(x: float) -> str:
    """68 -> "68th", 61 -> "61st", 100 -> "100th"; rounds first, so 99.6 reads "100th"."""
    n = int(round(x))
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _band_sentence(rc: dict, rk: dict) -> str:
    """How the shaded band of the equity chart was built, where the corrected run's final equity falls in it, and
    how that differs from the percentile row 9 reports; empty when the audit ran with no controls."""
    band = rc["band_series"]
    if len(band) == 0:
        return ""
    finals = np.cumprod(1.0 + band, axis=1)[:, -1]
    corrected_final = np.cumprod(1.0 + rk["series"]["corrected"])[-1]
    rank = (finals < corrected_final).mean() * 100.0
    lo, hi = BAND_QUANTILES
    return (f" The shaded band is, bar by bar, the {_ordinal(lo)} to {_ordinal(hi)} percentile of the equity of the {rc['controls']} matched random-entry runs of row 9, "
            f"charged the same {rc['cost_bps']:.0f} bps as the corrected run; the corrected run's final equity sits at the {_ordinal(rank)} percentile of theirs. "
            f"Row 9 ranks the cost-free Sharpe ratio instead, where the strategy sits at the {_ordinal(rc['percentile'])}: two statistics over the same runs.")


def run_audit(prices: pd.DataFrame, grid: list[Params], n_controls: int = 300, cost_bps: float = 20.0, universe: list[pd.DataFrame] | None = None) -> dict:
    """Run the naive backtest, then every audit step on the parameters the naive grid picked.

    The survivorship step needs a universe of symbols; without one the key is absent and the report has ten rows.
    """
    from naive_backtest import best_of_grid, one_sided_pvalue

    ranked = best_of_grid(prices)
    naive_sharpe, params, naive_trades, naive_summary = ranked[0]
    t_naive, p_naive = one_sided_pvalue(prices, naive_trades)
    result = {
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
        "control": steps.random_control(prices, params, n_controls=n_controls, cost_bps=cost_bps),
        "sensitivity": steps.parameter_sensitivity(prices, grid, params),
        "risk": steps.risk_profile(prices, params, cost_bps),
    }
    if universe is not None:
        result["survivorship"] = steps.survivorship(universe, params)
    return result


def render(r: dict, source: str, synthetic: bool = True, charts: dict[str, str] | None = None) -> str:
    p = r["params"]
    nv, la, sb, co, wi, ba, ts, mt, wf, rc, se, rk = (r[k] for k in ("naive", "lookahead", "same_bar", "costs", "windows", "baseline", "two_sided", "multiple", "walk_forward", "control", "sensitivity", "risk"))
    sv = r.get("survivorship")
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
        f"| 9 | No random-entry control | strategy Sharpe {_f(rc['strategy_sharpe'])} | {rc['controls']} matched random-entry runs: mean {_f(rc['control_mean'])}, sd {_f(rc['control_sd'])}; strategy at the {_ordinal(rc['percentile'])} percentile | {'material' if rc['percentile'] < 95 else 'minor'} |",
        f"| 10 | Only the best parameter point is reported; the surface around it is hidden | reported variant Sharpe {_f(se['reported_sharpe'])} (rank {se['rank']} of {len(se['surface'])}) | its {se['neighbours']} one-step neighbours: mean {_f(se['neighbour_mean'])} (min {_f(se['neighbour_min'])}, max {_f(se['neighbour_max'])}), {se['within_0_3']} of {se['neighbours']} within 0.3 of the reported value; grid median {_f(se['grid_median'])} | {'material' if _spike(se) else 'minor'} |",
        *([f"| 11 | Universe taken as of today: {sv['delisted']} of {sv['n_symbols']} symbols that fell below {sv['floor']:.0%} of their start were dropped from the history | survivors only ({sv['survivors']} symbols): mean per-symbol Sharpe {_f(sv['survivors_mean_symbol_sharpe'])}, symbols' mean buy-and-hold return {_pct(sv['survivors_mean_price_return'])} | point-in-time universe ({sv['n_symbols']} symbols, each until its delisting): mean per-symbol Sharpe {_f(sv['point_in_time_mean_symbol_sharpe'])}, symbols' mean buy-and-hold return {_pct(sv['all_mean_price_return'])} | {_sev(sv['survivors_mean_symbol_sharpe'] - sv['point_in_time_mean_symbol_sharpe'])} |"] if sv else []),
        "",
        "Rows 1-3 are corrections to the backtest itself and compound: after all three the honest Sharpe is "
        f"{_f(honest_sharpe)}. Rows 4-10 are corrections to the evidence: they do not change the returns, they change how much the returns prove. "
        "Row 10 reads the parameter surface: a plateau says the result does not hinge on one grid point, not that it is real; every variant on the plateau carries the same artifacts as rows 1-3."
        + (f" Row 11 corrects the data, not the evidence: honest rules, no cost, on a separate {sv['n_symbols']}-symbol synthetic universe, so its Sharpe is not comparable with the {_f(honest_sharpe)} of rows 1-3; "
           f"on the as-of-today universe the surviving symbols carry {(sv['survivors_mean_price_return'] - sv['all_mean_price_return']) * 100:.1f} points more buy-and-hold return than the point-in-time universe, and this strategy captures little of it." if sv else ""),
        "",
        "## Risk and exposure, claim vs corrected",
        "",
        "| Metric | Claim (as submitted) | Corrected (rows 1-3 applied) |",
        "|---|---|---|",
        f"| Max drawdown | {_pct(cl['max_drawdown'])} | {_pct(cr['max_drawdown'])} |",
        f"| Longest time under water | {_under_water(cl)} | {_under_water(cr)} |",
        f"| Annualised volatility | {_pct(cl['ann_vol'])} | {_pct(cr['ann_vol'])} |",
        f"| Exposure (share of bars in a position) | {_pct(cl['exposure'])} | {_pct(cr['exposure'])} |",
        f"| Turnover (trades per year) | {_f(cl['trades_per_year'], 1)} | {_f(cr['trades_per_year'], 1)} |",
        f"| Worst day | {_pct(cl['worst_day'])} | {_pct(cr['worst_day'])} |",
        f"| Expected shortfall, worst 5% of days (flat days count as 0) | {_pct(cl['es_5'])} | {_pct(cr['es_5'])} |",
        "",
        f"The corrected column is the same strategy with look-ahead removed, ambiguous bars booked to the stop and {co['cost_bps']:.0f} bps charged per round trip; "
        "it is what the risk figures would have been had the backtest been built correctly.",
        "",
    ]
    if charts is not None:
        lines += [
            "## Charts",
            "",
            f"![Equity and drawdown: claim, look-ahead removed only, corrected]({charts['equity']})",
            "",
            f"![Naive Sharpe across the parameter grid at the chosen stop]({charts['sensitivity']})",
            "",
            "Both charts are generated by `python -m audit` as plain SVG; the equity panel compounds the daily mark-to-market returns from 1.0 on a log axis, "
            "the lower panel is the drawdown from the running peak. The dashed curve is the claim with only the look-ahead removed (row 1): the gap between it and the claim "
            f"is what knowing the close before entering was worth, and the gap to the corrected curve is rows 2 and 3.{_band_sentence(rc, rk)} "
            f"The heatmap is the stop = {p.stop_atr} ATR slice of the grid; the neighbours along the stop axis are not in it.",
            "",
        ]
    lines += ["## Material weaknesses", ""]
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
        *(["- Build the universe point-in-time: every symbol that existed at the decision time, including those later delisted, with positions closed at the delisting price."] if sv else []),
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
        *(["- The universe construction rule with the delisting source and date for every symbol that left the sample."] if sv else []),
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


def _spike(se: dict) -> bool:
    """The reported point is a spike when its one-step neighbours average less than half of it; otherwise it sits on a plateau.

    Only a positive reported Sharpe can be a spike, and the gap has to be material in absolute terms too: half of almost
    nothing, or a negative best, says nothing about the surface.
    """
    r, m = se["reported_sharpe"], se["neighbour_mean"]
    return r > 0 and m < 0.5 * r and r - m >= 0.3


def _under_water(p: dict) -> str:
    return f"{p['drawdown_bars']} bars" + (" (not recovered)" if p["not_recovered"] else "")


def _sev_t(t_before: float, t_after: float) -> str:
    """Severity of an evidence artifact: it matters when it manufactures significance."""
    if abs(t_before) >= 1.96 > abs(t_after):
        return "material: significance disappears"
    if abs(t_before) >= 1.96 and abs(t_before) - abs(t_after) >= 1.0:
        return "material"
    return "minor"


def _weaknesses(r: dict) -> list[str]:
    la, sb, co, wi, mt, wf, rc, se, rk = (r[k] for k in ("lookahead", "same_bar", "costs", "windows", "multiple", "walk_forward", "control", "sensitivity", "risk"))
    sv = r.get("survivorship")
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
    if _spike(se):
        out.append(f"The reported parameters sit on a spike: their one-step neighbours average a Sharpe of {_f(se['neighbour_mean'])} against the reported {_f(se['reported_sharpe'])}, so the result depends on one point of the grid.")
    if sv and sv["survivors_mean_symbol_sharpe"] - sv["point_in_time_mean_symbol_sharpe"] >= 0.3:
        out.append(f"The as-of-today universe removes the {sv['delisted']} symbols that failed; survivors alone show a mean per-symbol Sharpe of {_f(sv['survivors_mean_symbol_sharpe'])} against {_f(sv['point_in_time_mean_symbol_sharpe'])} on the point-in-time universe.")
    if rk["corrected"]["max_drawdown"] > rk["claim"]["max_drawdown"]:
        out.append(f"The corrected run's maximum drawdown is {_pct(rk['corrected']['max_drawdown'])} against {_pct(rk['claim']['max_drawdown'])} in the claim.")
    return out or [f"No material weakness found by the {'eleven' if sv else 'ten'} checks."]
