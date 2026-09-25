"""Download daily BTC/USDT candles from Binance's public REST API into data/btc_daily.csv.

No API key is needed. The CSV is committed so the appendix runs offline; re-run this script to refresh it.
"""

from __future__ import annotations

import datetime as dt
import json
import pathlib
import urllib.request

URL = "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1d&limit=1000&startTime={start}"
FIRST_DAY_MS = 1502928000000  # 2017-08-17, first BTCUSDT daily candle
HERE = pathlib.Path(__file__).resolve().parent


def fetch_all() -> list[list]:
    rows, start = [], FIRST_DAY_MS
    while True:
        with urllib.request.urlopen(URL.format(start=start), timeout=60) as resp:
            batch = json.load(resp)
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < 1000:
            break
        start = batch[-1][6] + 1  # next candle after this batch's close time
    return rows


def main() -> None:
    rows = fetch_all()
    lines = ["Date,Open,High,Low,Close,Volume"]
    for r in rows[:-1]:  # drop the still-open candle of today
        day = dt.datetime.fromtimestamp(r[0] / 1000, dt.timezone.utc).date().isoformat()
        lines.append(f"{day},{float(r[1]):.2f},{float(r[2]):.2f},{float(r[3]):.2f},{float(r[4]):.2f},{float(r[5]):.4f}")
    (HERE / "btc_daily.csv").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    (HERE / "FETCHED.txt").write_text(f"{URL.split('?')[0]} symbol=BTCUSDT interval=1d\nfetched {dt.datetime.now(dt.timezone.utc).date().isoformat()} UTC, {len(lines) - 1} rows, {lines[1][:10]} to {lines[-1][:10]}\n", encoding="utf-8")
    print(f"wrote btc_daily.csv ({len(lines) - 1} rows, {lines[1][:10]} to {lines[-1][:10]})")


if __name__ == "__main__":
    main()
