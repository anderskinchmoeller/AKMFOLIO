#!/usr/bin/env python3
"""
Fetch 2025-12-26 -> 2026-09-01 daily prices from Yahoo Finance for the tickers
that were in AKMFOLIO's investable universe as of the last WRDS week
(2025-12-26), and compress them into weekly (Friday-ending) returns that can
be appended to data/weekly_returns.csv.

Run this from a REGULAR terminal on your Mac (not through the Claude bridge -
that sandbox blocks general internet access). From the repo root:

    cd ~/AKMFOLIO
    pip3 install --upgrade yfinance
    python3 fetch_yfinance_extension.py

It is resumable: progress is checkpointed to data/_yfinance_raw_prices.csv
after every batch, so if it's interrupted (Ctrl-C, network hiccup, rate
limit) you can just re-run it and it will skip tickers it already has.

Outputs (both under data/):
  - _yfinance_raw_prices.csv      wide matrix of daily Adj Close, permno columns
  - yfinance_extension_returns.csv  weekly (W-FRI) compounded returns, permno columns
  - yfinance_extension_report.json  what worked, what didn't, date coverage

Nothing in data/weekly_returns.csv itself is touched by this script. Once
it's done, tell Claude and it will merge yfinance_extension_returns.csv into
the real weekly_returns.csv (with a backup taken first).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

try:
    import yfinance as yf
except ImportError:
    print("Install yfinance first:  pip3 install --upgrade yfinance", file=sys.stderr)
    raise

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"
TICKER_LIST_PATH = DATA_DIR / "_yfinance_ticker_list.csv"
RAW_PRICES_PATH = DATA_DIR / "_yfinance_raw_prices.csv"
OUTPUT_RETURNS_PATH = DATA_DIR / "yfinance_extension_returns.csv"
REPORT_PATH = DATA_DIR / "yfinance_extension_report.json"

# One day before the first new trading day, so pct_change() has a baseline.
START = "2025-12-26"
# yfinance end is exclusive; this captures through 2026-09-01.
END = "2026-09-02"

BATCH_SIZE = 75
SLEEP_BETWEEN_BATCHES_SEC = 2.0
MAX_RETRIES_PER_BATCH = 3


def load_ticker_map() -> pd.DataFrame:
    if not TICKER_LIST_PATH.exists():
        raise SystemExit(
            f"Missing {TICKER_LIST_PATH}. This should have been generated "
            "alongside this script -- ask Claude to regenerate it if it's gone."
        )
    df = pd.read_csv(TICKER_LIST_PATH, dtype={"permno": "int64", "ticker": "string"})
    df = df.dropna(subset=["ticker"])
    df["ticker"] = df["ticker"].str.strip()
    df = df[df["ticker"] != ""]
    # yfinance uses '-' for class shares where CRSP tickers often use nothing
    # special (e.g. BRK.A vs BRK-A); try the yfinance convention.
    df["yf_ticker"] = df["ticker"].str.replace(".", "-", regex=False)
    return df.drop_duplicates(subset=["yf_ticker"])


def load_checkpoint() -> pd.DataFrame:
    if RAW_PRICES_PATH.exists():
        return pd.read_csv(RAW_PRICES_PATH, index_col=0, parse_dates=True)
    return pd.DataFrame()


def save_checkpoint(prices: pd.DataFrame) -> None:
    tmp = RAW_PRICES_PATH.with_suffix(".csv.tmp")
    prices.to_csv(tmp)
    tmp.replace(RAW_PRICES_PATH)


def fetch_batch(tickers: list[str]) -> pd.DataFrame:
    last_exc = None
    for attempt in range(1, MAX_RETRIES_PER_BATCH + 1):
        try:
            raw = yf.download(
                tickers,
                start=START,
                end=END,
                auto_adjust=True,  # dividend/split-adjusted Close
                progress=False,
                threads=True,
                group_by="ticker",
            )
            if raw.empty:
                return pd.DataFrame()
            if len(tickers) == 1:
                # yfinance doesn't multi-index a single-ticker download.
                close = raw[["Close"]].rename(columns={"Close": tickers[0]})
            else:
                close = pd.DataFrame(
                    {t: raw[t]["Close"] for t in tickers if t in raw.columns.get_level_values(0)}
                )
            return close
        except Exception as exc:  # noqa: BLE001 - genuinely want to retry+report anything
            last_exc = exc
            print(f"  batch failed (attempt {attempt}/{MAX_RETRIES_PER_BATCH}): {exc}")
            time.sleep(5 * attempt)
    print(f"  giving up on this batch: {last_exc}")
    return pd.DataFrame()


def main() -> None:
    ticker_map = load_ticker_map()
    all_tickers = ticker_map["yf_ticker"].tolist()
    print(f"{len(all_tickers)} tickers to fetch, {START} -> {END}")

    prices = load_checkpoint()
    already = set(prices.columns)
    remaining = [t for t in all_tickers if t not in already]
    print(f"{len(already)} already fetched, {len(remaining)} remaining")

    failed: list[str] = []
    for i in range(0, len(remaining), BATCH_SIZE):
        batch = remaining[i : i + BATCH_SIZE]
        print(f"batch {i // BATCH_SIZE + 1}/{-(-len(remaining) // BATCH_SIZE)}: {batch[0]}..{batch[-1]}")
        close = fetch_batch(batch)
        got = set(close.columns) if not close.empty else set()
        failed.extend(t for t in batch if t not in got)
        if not close.empty:
            prices = close.combine_first(prices) if prices.empty else prices.combine_first(close)
            prices = prices.combine_first(close)
            save_checkpoint(prices)
        time.sleep(SLEEP_BETWEEN_BATCHES_SEC)

    if prices.empty:
        raise SystemExit("Nothing was fetched -- check your internet connection and retry.")

    prices = prices.sort_index()
    prices.index = pd.to_datetime(prices.index)

    # Weekly (W-FRI) compounded return from daily simple returns.
    daily_ret = prices.pct_change()
    week = daily_ret.index.to_series().dt.to_period("W-FRI").dt.end_time.dt.normalize()
    grouped = daily_ret.groupby(week)
    weekly_ret = grouped.apply(lambda g: (1.0 + g).prod() - 1.0)

    # Drop an incomplete trailing week (market data for that week isn't
    # finished yet), mirroring the CRSP build's drop_incomplete_final_week.
    last_trading_day = prices.index.max()
    if weekly_ret.index[-1] > last_trading_day:
        # can't happen given how 'week' is derived, but guard anyway
        weekly_ret = weekly_ret.iloc[:-1]
    last_week_end = weekly_ret.index[-1]
    if last_week_end.date().isoweekday() == 5 and last_trading_day < last_week_end - pd.Timedelta(days=1):
        weekly_ret = weekly_ret.iloc[:-1]

    # First row is NaN (pct_change baseline) once resampled into its week --
    # drop that week too since it mixes the pre-existing 2025-12-26 week.
    weekly_ret = weekly_ret.loc[weekly_ret.index > pd.Timestamp(START)]

    # Map yf_ticker -> permno for the output columns.
    yf_to_permno = dict(zip(ticker_map["yf_ticker"], ticker_map["permno"]))
    weekly_ret = weekly_ret.rename(columns=yf_to_permno)
    weekly_ret = weekly_ret.loc[:, weekly_ret.columns.notna()]
    weekly_ret.columns = weekly_ret.columns.astype("int64").astype(str)
    weekly_ret.index.name = "date"

    weekly_ret.to_csv(OUTPUT_RETURNS_PATH)

    report = {
        "start_requested": START,
        "end_requested": END,
        "n_tickers_requested": len(all_tickers),
        "n_tickers_fetched": len(prices.columns),
        "n_tickers_failed": len(failed),
        "failed_tickers": sorted(failed),
        "last_trading_day_seen": str(last_trading_day.date()),
        "weeks_produced": [str(d.date()) for d in weekly_ret.index],
        "note": (
            "Adjusted-close daily returns from Yahoo Finance, NOT WRDS CRSP "
            "DlyRet. No delisting-return reconciliation. Ticker-based mapping "
            "(some CRSP permnos may not match a live Yahoo symbol if the "
            "ticker changed)."
        ),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2))

    print(f"\nDone. {len(weekly_ret)} new week(s): {list(weekly_ret.index.date)}")
    print(f"Covered {len(prices.columns)}/{len(all_tickers)} tickers ({len(failed)} failed).")
    print(f"-> {OUTPUT_RETURNS_PATH}")
    print(f"-> {REPORT_PATH}")
    print("\nTell Claude you're done and it will merge this into weekly_returns.csv.")


if __name__ == "__main__":
    main()
