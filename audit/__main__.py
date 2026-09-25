"""Run the full audit and write AUDIT_REPORT.md.

  python -m audit                       # synthetic data, seed 7, 3000 bars
  python -m audit --csv data/btc_daily.csv --out APPENDIX_BTC.md
"""

from __future__ import annotations

import argparse
import sys
import time

from audit.data import load_csv, synthetic_prices
from audit.report import render, run_audit


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m audit")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--bars", type=int, default=3000)
    ap.add_argument("--csv", help="daily OHLC CSV instead of synthetic data")
    ap.add_argument("--controls", type=int, default=300, help="random-entry control runs")
    ap.add_argument("--cost-bps", type=float, default=20.0)
    ap.add_argument("--out", default="AUDIT_REPORT.md")
    args = ap.parse_args(argv)

    from naive_backtest import GRID

    if args.csv:
        prices, source = load_csv(args.csv), f"`{args.csv}`"
    else:
        prices, source = synthetic_prices(args.bars, args.seed), f"synthetic driftless random walk, seed {args.seed}"
    t0 = time.perf_counter()
    result = run_audit(prices, GRID, n_controls=args.controls, cost_bps=args.cost_bps)
    text = render(result, source)
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    nv, co, rc = result["naive"], result["costs"], result["control"]
    print(f"claimed Sharpe {nv['sharpe']:.2f} -> honest Sharpe {co['after']['sharpe']:.2f}; random-entry percentile {rc['percentile']:.0f}; report: {args.out} ({time.perf_counter() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
