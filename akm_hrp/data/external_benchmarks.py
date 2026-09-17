"""Buy-and-hold external benchmarks (e.g. URTH, iShares MSCI World ETF).

An external benchmark is not an allocator: it holds one fund and never
rebalances, so it is added to the comparison as a ready-made return series
rather than walked forward over the stock universe. Returns are weekly total
returns (dividend-adjusted close, compounded to Friday) aligned to the weekly
return panel's Friday labels. Weeks before the fund existed are NaN, so every
metric and active-return test only uses the fund's live history.

Prices come from a cached CSV (``<dir>/<TICKER>_weekly_returns.csv``). When
the cache is missing it is fetched once with yfinance (``pip install
yfinance``) and written there; or pre-fetch it with::

    python -m akm_hrp.data.external_benchmarks URTH --output-dir data/benchmarks
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from akm_hrp.backtest.engine import BacktestResult

BENCHMARK_DESCRIPTIONS = {
    "URTH": "iShares MSCI World ETF",
    "ACWI": "iShares MSCI ACWI ETF",
    "SPY": "SPDR S&P 500 ETF",
}


def benchmark_cache_path(ticker: str, directory: str | Path) -> Path:
    return Path(directory) / f"{ticker.upper()}_weekly_returns.csv"


def weekly_returns_from_prices(prices: pd.Series, week_frequency: str = "W-FRI") -> pd.Series:
    """Compound daily adjusted closes into week-ending total returns."""

    prices = pd.to_numeric(prices, errors="coerce").dropna()
    prices = prices[prices > 0].sort_index()
    prices.index = pd.DatetimeIndex(prices.index).tz_localize(None).normalize()
    weekly_close = prices.resample(week_frequency).last().dropna()
    returns = weekly_close.pct_change()
    # The first week has no prior close: its return is unknown, not zero.
    return returns.iloc[1:].rename("return")


def fetch_weekly_returns(ticker: str, start: str = "1990-01-01") -> pd.Series:
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            f"No cached returns for {ticker} and yfinance is not installed. "
            "Run `pip install yfinance` or supply the CSV."
        ) from exc
    frame = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if frame is None or frame.empty:
        raise RuntimeError(f"yfinance returned no prices for {ticker}.")
    close = frame["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return weekly_returns_from_prices(close)


def load_benchmark_returns(
    ticker: str,
    directory: str | Path = "data/benchmarks",
    *,
    fetch_if_missing: bool = True,
) -> pd.Series:
    path = benchmark_cache_path(ticker, directory)
    if path.exists():
        frame = pd.read_csv(path, index_col=0, parse_dates=True)
        series = pd.to_numeric(frame.iloc[:, 0], errors="coerce")
    elif fetch_if_missing:
        series = fetch_weekly_returns(ticker)
        path.parent.mkdir(parents=True, exist_ok=True)
        series.to_frame("return").to_csv(path, index_label="date")
        print(f"saved {ticker} weekly returns: {path.resolve()}", flush=True)
    else:
        raise FileNotFoundError(path)
    series.index = pd.DatetimeIndex(series.index).normalize()
    return series.sort_index().rename(ticker.upper())


def benchmark_backtest_result(
    ticker: str,
    returns: pd.Series,
    index: pd.Index,
) -> BacktestResult:
    """Wrap a buy-and-hold return series in the walk-forward result type."""

    aligned = returns.reindex(pd.DatetimeIndex(index))
    if aligned.notna().sum() == 0:
        raise ValueError(
            f"{ticker} returns do not overlap the backtest dates "
            f"({index[0]:%Y-%m-%d} .. {index[-1]:%Y-%m-%d})."
        )
    # A weekly label inside the fund's life with no price (market closure) is
    # a flat week for a buy-and-hold holder, not a hole in the sample. Weeks
    # before inception or after the cached data ends stay NaN.
    first, last = aligned.first_valid_index(), aligned.last_valid_index()
    aligned.loc[first:last] = aligned.loc[first:last].fillna(0.0)
    if last < aligned.index[-1]:
        print(
            f"warning: {ticker} returns end {last:%Y-%m-%d} but the backtest runs to "
            f"{aligned.index[-1]:%Y-%m-%d}; refresh the cache to extend it.",
            flush=True,
        )
    held = aligned.notna().astype(float)
    weights = pd.DataFrame({ticker.upper(): held}, index=aligned.index)
    zeros = pd.Series(0.0, index=aligned.index)
    return BacktestResult(
        weights=weights,
        portfolio_returns=aligned.rename("portfolio_return"),
        turnover=zeros.rename("turnover"),
        metrics={},
        transaction_costs=zeros.rename("transaction_cost"),
        ending_weights=pd.Series({ticker.upper(): 1.0}),
        validation_mode="external_buy_and_hold",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache weekly benchmark ETF returns.")
    parser.add_argument("tickers", nargs="+")
    parser.add_argument("--output-dir", default="data/benchmarks")
    args = parser.parse_args()
    for ticker in args.tickers:
        series = fetch_weekly_returns(ticker)
        path = benchmark_cache_path(ticker, args.output_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        series.to_frame("return").to_csv(path, index_label="date")
        print(
            f"{ticker}: {len(series)} weeks {series.index[0]:%Y-%m-%d} .. "
            f"{series.index[-1]:%Y-%m-%d}, ann. return "
            f"{(1 + series).prod() ** (52 / len(series)) - 1:.2%} -> {path}"
        )


if __name__ == "__main__":
    main()
