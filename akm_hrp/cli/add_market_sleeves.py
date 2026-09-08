from __future__ import annotations

"""Download liquid market sleeves and append them to a balanced HRP bundle."""

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

DEFAULT_TICKERS = (
    "GLD",
    "TIP",
    "TLT",
    "DBMF",
    "BTC-USD",
    "DBC",
    "UUP",
    "EEM",
    "EFA",
    "USO",
    "VNQ",
)

ASSET_ROLES = {
    "GLD": ("macro:gold", "gold bullion ETF"),
    "TIP": ("macro:inflation_linked_bond", "US TIPS ETF"),
    "TLT": ("macro:long_duration_bond", "long Treasury ETF"),
    "DBMF": ("macro:managed_futures", "managed-futures ETF"),
    "BTC-USD": ("macro:digital_asset", "bitcoin USD spot proxy"),
    "DBC": ("macro:broad_commodities", "broad commodity ETF"),
    "UUP": ("macro:US_dollar", "US dollar ETF"),
    "EEM": ("macro:emerging_equity", "emerging-markets equity ETF"),
    "EFA": ("macro:developed_ex_US_equity", "developed ex-US equity ETF"),
    "USO": ("macro:oil", "oil-futures ETF"),
    "VNQ": ("macro:real_estate", "US REIT ETF"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download adjusted daily prices, form Friday weekly returns, and "
            "append PIT-safe sleeves to a balanced HRP bundle."
        )
    )
    parser.add_argument(
        "--bundle-dir",
        default="wrds_top2500/balanced_hrp",
        help="Directory containing weekly_returns.csv and pit_universe.csv.",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=list(DEFAULT_TICKERS),
        help="Yahoo Finance symbols to add.",
    )
    parser.add_argument(
        "--minimum-history-weeks",
        type=int,
        default=52,
        help="Activate each sleeve after this many observed weekly returns.",
    )
    return parser.parse_args()


def _load_wide(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    frame.columns = frame.columns.astype(str)
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError(f"{path} must have dates in its first column.")
    if frame.index.has_duplicates:
        raise ValueError(f"{path} contains duplicate dates.")
    return frame.apply(pd.to_numeric, errors="coerce")


def _download_adjusted_close(
    tickers: list[str], start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise RuntimeError("Install yfinance before running this command.") from exc

    data = yf.download(
        tickers,
        start=start.date().isoformat(),
        end=(end + pd.Timedelta(days=1)).date().isoformat(),
        auto_adjust=True,
        actions=False,
        group_by="column",
        progress=False,
        threads=True,
    )
    if data.empty:
        raise RuntimeError("The market-data download returned no observations.")
    if isinstance(data.columns, pd.MultiIndex):
        if "Close" not in data.columns.get_level_values(0):
            raise RuntimeError("The download did not contain adjusted close prices.")
        close = data["Close"].copy()
    else:
        if "Close" not in data:
            raise RuntimeError("The download did not contain an adjusted close price.")
        close = data[["Close"]].rename(columns={"Close": tickers[0]})
    close.columns = close.columns.astype(str)
    missing = sorted(set(tickers).difference(close.columns))
    if missing:
        raise RuntimeError(f"No downloaded price column for: {missing}")
    close = close.reindex(columns=tickers).apply(pd.to_numeric, errors="coerce")
    unusable = [ticker for ticker in tickers if close[ticker].notna().sum() < 2]
    if unusable:
        raise RuntimeError(f"Too few adjusted prices for: {unusable}")
    return close


def _weekly_returns(
    adjusted_close: pd.DataFrame, weekly_index: pd.DatetimeIndex
) -> pd.DataFrame:
    weekly_prices = adjusted_close.resample("W-FRI").last()
    weekly = weekly_prices.pct_change(fill_method=None)
    weekly = weekly.reindex(weekly_index)
    if (weekly < -1.0 - 1e-12).any().any():
        raise ValueError("A downloaded weekly return is below -100%.")
    for ticker in weekly:
        valid = weekly[ticker].notna()
        interior_gap = (~valid) & valid.cummax() & valid.iloc[::-1].cummax().iloc[::-1]
        if interior_gap.any():
            dates = weekly.index[interior_gap][:5].date.tolist()
            raise ValueError(f"{ticker} has interior weekly gaps near {dates}.")
    return weekly


def _pit_from_returns(returns: pd.DataFrame, minimum_history: int) -> pd.DataFrame:
    if minimum_history < 2:
        raise ValueError("--minimum-history-weeks must be at least 2.")
    valid = returns.notna()
    return (valid & (valid.cumsum() >= minimum_history)).astype("int8")


def _backup_once(path: Path) -> Path:
    backup = path.with_name(f"{path.stem}.before_market_sleeves{path.suffix}")
    if not backup.exists():
        shutil.copy2(path, backup)
    return backup


def _atomic_csv(frame: pd.DataFrame, path: Path, *, index: bool = True) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=index)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    bundle = Path(args.bundle_dir)
    returns_path = bundle / "weekly_returns.csv"
    pit_path = bundle / "pit_universe.csv"
    metadata_path = bundle / "crsp_security_metadata.csv"
    manifest_path = bundle / "universe_manifest.json"
    required = [returns_path, pit_path, metadata_path, manifest_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Balanced bundle is missing: " + ", ".join(missing))

    tickers = [
        "DBMF" if ticker.upper() == "DBFM" else ticker.upper()
        for ticker in args.tickers
    ]
    tickers = list(dict.fromkeys(tickers))
    unsupported = sorted(set(tickers).difference(ASSET_ROLES))
    if unsupported:
        raise ValueError(f"No asset-role metadata configured for: {unsupported}")

    returns = _load_wide(returns_path)
    pit = _load_wide(pit_path).fillna(0).astype("int8")
    if not returns.index.equals(pit.index):
        raise ValueError("Balanced returns and PIT masks use different dates.")
    overlap = returns.columns.intersection(tickers)
    if len(overlap):
        print(f"updating existing sleeves: {overlap.tolist()}", flush=True)

    prices = _download_adjusted_close(
        tickers,
        returns.index.min() - pd.Timedelta(days=10),
        returns.index.max(),
    )
    sleeves = _weekly_returns(prices, returns.index)
    sleeve_pit = _pit_from_returns(sleeves, args.minimum_history_weeks)
    combined_returns = pd.concat(
        [returns.drop(columns=tickers, errors="ignore"), sleeves], axis=1
    )
    combined_pit = pd.concat(
        [pit.drop(columns=tickers, errors="ignore"), sleeve_pit], axis=1
    ).astype("int8")

    metadata = pd.read_csv(metadata_path, dtype=str)
    metadata = metadata.loc[~metadata["asset"].isin(tickers)].copy()
    sleeve_metadata = pd.DataFrame(
        [
            {
                "asset": ticker,
                "permno": "",
                "ticker": ticker,
                "cluster": ASSET_ROLES[ticker][0],
                "role": ASSET_ROLES[ticker][1],
            }
            for ticker in tickers
        ]
    )
    metadata = pd.concat([metadata, sleeve_metadata], ignore_index=True)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    macro_columns = list(dict.fromkeys([*manifest.get("macro_columns", []), *tickers]))
    manifest["macro_columns"] = macro_columns
    manifest["n_assets"] = int(combined_returns.shape[1])
    manifest["market_sleeve_source"] = "Yahoo Finance adjusted close"
    manifest["market_sleeve_symbol_aliases"] = {
        "GOLD": "GLD",
        "DBFM": "DBMF",
        "BITCOIN": "BTC-USD",
    }
    manifest["market_sleeve_minimum_history_weeks"] = int(
        args.minimum_history_weeks
    )

    backups = [_backup_once(path) for path in required]
    _atomic_csv(combined_returns, returns_path)
    _atomic_csv(combined_pit, pit_path)
    _atomic_csv(metadata, metadata_path, index=False)
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_manifest.replace(manifest_path)
    _atomic_csv(prices, bundle / "market_sleeve_adjusted_close.csv")

    print(f"saved: {returns_path.resolve()}")
    print(f"saved: {pit_path.resolve()}")
    print(f"saved: {metadata_path.resolve()}")
    print(f"added sleeves: {tickers}")
    print("backups: " + ", ".join(str(path.resolve()) for path in backups))
    for ticker in tickers:
        active = combined_pit.index[combined_pit[ticker].astype(bool)]
        first = active.min().date().isoformat() if len(active) else "not active"
        observations = int(sleeves[ticker].notna().sum())
        print(f"{ticker}: {observations} weekly returns; PIT active from {first}")


if __name__ == "__main__":
    main()
