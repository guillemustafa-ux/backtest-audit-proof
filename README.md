# backtest-audit-proof

A backtest that finds a Sharpe ratio of 3.86 and a p-value of 10⁻⁵¹ in data that has **no edge by construction**, and the audit that takes that claim apart one methodological artifact at a time, ending in the written review a research-methodology audit is expected to deliver.

The point is not the strategy. The point is that every number in a backtest report comes from a chain of decisions, and each decision can manufacture performance. On a driftless random walk the truth is known: whatever the naive backtest finds is an artifact, so the audit can put a number on each one.

## Run it

```bash
python -m venv .venv && .venv/Scripts/pip install -e ".[test]"   # Windows; on macOS/Linux use .venv/bin/pip
python naive_backtest.py          # the claim: Sharpe 3.86, win rate 93.6%, p = 5.8e-51
python -m audit                   # the audit: writes AUDIT_REPORT.md and charts/ in about five seconds
python -m pytest                  # 51 tests; the artifact tests assert each artifact moves its metric the predicted way
python -m audit --csv data/btc_daily.csv --out APPENDIX_BTC.md   # same audit on real BTC daily data, offline
```

Same seed, same numbers: `AUDIT_REPORT.md` and the charts regenerate byte-identical.

## What the audit checks

| # | Artifact planted in the naive backtest | Correction | Synthetic result |
|---|---|---|---|
| 1 | The signal uses the close of the bar it enters on (look-ahead) | Enter at the next bar's open | Sharpe 3.86 → 0.13 |
| 2 | Target and stop both hit in one bar, booked as a win | Book the stop | 0.13 → −0.19 (8 of 230 exits) |
| 3 | Zero transaction costs | 20 bps round trip | −0.19 → −0.62 |
| 4 | Overlapping forward-return windows counted as independent | Newey-West and block-bootstrap t, effective N | N 273 → 139 |
| 5 | Event returns tested against zero | Test against the unconditional baseline, with Newey-West standard errors because those windows overlap too | on BTC: t 3.61 → 1.28 (i.i.d.) → 0.97 (Newey-West) |
| 6 | Direction of the test chosen after seeing the sign | Two-sided test | p doubles |
| 7 | Best of 125 parameter variants reported as *the* result | Benjamini-Hochberg FDR, deflated Sharpe ratio | 6 "significant" → 0; noise alone expects a best Sharpe of 2.39 |
| 8 | Parameters optimized on the whole sample | Walk-forward with a holdout touched once | 0.54 in-sample → 0.29 holdout |
| 9 | No control | Matched random-entry control: same trade count, holding periods and stop width | strategy at the 68th percentile of 300 random runs |
| 10 | Only the best parameter point is reported; the surface around it is hidden | Report the one-step neighbourhood and the grid distribution next to the chosen variant | reported 3.86 vs its 4 neighbours averaging 3.43 (min 3.17, max 3.77), 2 of 4 within 0.3; grid median 1.65; a plateau, not a spike |
| 11 | Universe built as of today: the symbols that failed are missing from the history (survivorship) | Point-in-time universe: every symbol until its delisting, positions closed at the delisting close | 17 of 40 synthetic symbols fell below 50% of their start; survivors only: mean per-symbol Sharpe −0.08, symbols' mean buy-and-hold return 54.3%; point-in-time: −0.18 and 9.8% |

Rows 1–3 fix the backtest and compound. Rows 4–10 fix the evidence: they change how much the returns prove, not the returns. Row 10 reads the parameter surface: a plateau says the result does not hinge on one grid point, not that it is real; every variant on it carries the same artifacts as rows 1–3. Row 11 fixes the data: honest rules, no cost, on a separate 40-symbol synthetic universe, so its Sharpe is not comparable with the −0.62 of rows 1–3; the surviving symbols carry 44.5 points more buy-and-hold return than the point-in-time universe, and the strategy captures little of it.

The report ends with the four sections a methodology review is asked for: material weaknesses, required corrections, recommended methodology for the next pre-registered run, and the evidence required before confirmatory testing.

## Risk profile and charts

The report also carries a risk and exposure table for the claim and for the corrected run (rows 1–3 applied): maximum drawdown and the longest time under water, annualised volatility, exposure (share of bars in a position), turnover (trades per year), worst day and the 5% expected shortfall. On the synthetic data the claim's maximum drawdown is 5.7% and its longest time under water 116 bars; corrected, 52.6% and 2941 bars, not recovered by the end of the sample. Two charts are written next to the report as plain SVG, with no plotting dependency so they regenerate byte-identical: equity and drawdown of claim vs corrected, the equity on a log axis so both curves stay visible, and the naive Sharpe across the parameter grid at the chosen stop with the reported cell framed.

![Equity and drawdown, claim vs corrected](charts/AUDIT_REPORT_equity.svg)

![Naive Sharpe across the parameter grid at the chosen stop](charts/AUDIT_REPORT_sensitivity.svg)

## Layout

- `audit/data.py` — driftless random walk with fat-tailed intrabar wicks (Student t, 3 df), a universe of them for the survivorship row, and a CSV loader.
- `audit/engine.py` — long-only breakout engine; every artifact is a switch in `Rules`.
- `audit/steps.py` — the eleven audit steps and the risk profile, one function each.
- `audit/stats.py` — Newey-West, moving-block bootstrap, effective sample size, Benjamini-Hochberg, expected maximum Sharpe and deflated Sharpe ratio (Bailey & López de Prado, 2014), maximum drawdown and expected shortfall.
- `audit/charts.py` — the two charts as hand-written SVG.
- `audit/report.py` — renders the review; `python -m audit` runs everything.
- `naive_backtest.py` — the claim, with its 125-variant grid.
- `data/btc_daily.csv` — BTC/USDT daily candles from Binance's public API, committed with `data/FETCHED.txt` so the appendix runs offline; `data/fetch_btc.py` refreshes it.

Dependencies: pandas, numpy, scipy. No backtesting framework, on purpose: in an audit every line has to be readable.

## Limits, stated plainly

- Daily bars. Same-bar ambiguity (row 2) can only be resolved with intrabar data; here it is resolved conservatively, which is the right default and still an assumption.
- Long only, one position at a time, one instrument per run; the survivorship row averages 40 synthetic instruments equal-weight. The random control is matched per instrument as it should be; a multi-instrument study needs the control per instrument too.
- The survivorship row uses a synthetic universe of 40 random walks with a 50% delisting floor; it demonstrates the mechanism, not any real index.
- No regime breakdown: the audit does not split the sample by volatility or trend regime.
- The BTC appendix is on real data, where the truth is unknown. The audit there says how much of the claim survives each correction and how much the baseline drift explains — not that the edge is zero.
- This audits a backtest's methodology. It does not model options, intraday quotes, or execution latency.
