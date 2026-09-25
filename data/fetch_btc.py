"""Download daily BTC/USD OHLC from Stooq into data/btc_daily.csv and record the fetch date.

The CSV is committed so the appendix runs offline; re-run this script to refresh it.
"""

from __future__ import annotations

import datetime as dt
import pathlib
import urllib.request

URL = "https://stooq.com/q/d/l/?s=btcusd&i=d"
HERE = pathlib.Path(__file__).resolve().parent


def main() -> None:
    raw = urllib.request.urlopen(URL, timeout=60).read().decode("utf-8")
    if not raw.startswith("Date,Open,High,Low,Close"):
        raise SystemExit(f"unexpected payload: {raw[:80]!r}")
    (HERE / "btc_daily.csv").write_text(raw, encoding="utf-8", newline="\n")
    rows = raw.count("\n") - 1
    (HERE / "FETCHED.txt").write_text(f"{URL}\nfetched {dt.date.today().isoformat()} UTC, {rows} rows\n", encoding="utf-8")
    print(f"wrote btc_daily.csv ({rows} rows)")


if __name__ == "__main__":
    main()
