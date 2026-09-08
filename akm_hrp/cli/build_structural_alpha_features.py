from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

_CACHE_DATE_PATTERN = re.compile(r"_(\d{8})_(\d{8})_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build lagged weekly size/liquidity features from the CRSP daily cache."
        )
    )
    parser.add_argument("--daily-cache-dir", required=True)
    parser.add_argument(
        "--output",
        default="data/structural_alpha_features.csv.gz",
    )
    parser.add_argument("--start", type=pd.Timestamp, default=None)
    parser.add_argument("--end", type=pd.Timestamp, default=None)
    parser.add_argument("--rolling-days", type=int, default=20)
    parser.add_argument(
        "--market-cap-scale",
        type=float,
        default=1000.0,
        help="Scale CRSP DlyCap to dollars (default: 1000).",
    )
    return parser.parse_args()


def _file_start(path: Path) -> str:
    match = _CACHE_DATE_PATTERN.search(path.name)
    return match.group(1) if match else path.name


def _file_range(path: Path) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    match = _CACHE_DATE_PATTERN.search(path.name)
    if match is None:
        return None
    return pd.Timestamp(match.group(1)), pd.Timestamp(match.group(2))


def build_structural_features(
    daily_cache_dir: str | Path,
    *,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    rolling_days: int = 20,
    market_cap_scale: float = 1000.0,
) -> pd.DataFrame:
    """Build strictly lagged weekly market-cap and microstructure proxies."""
    if rolling_days < 2:
        raise ValueError("rolling_days must be at least two.")
    if market_cap_scale <= 0.0:
        raise ValueError("market_cap_scale must be positive.")
    directory = Path(daily_cache_dir)
    files = sorted(directory.glob("*.csv.gz"), key=_file_start)
    if not files:
        raise FileNotFoundError(f"No .csv.gz daily-cache files found in {directory}")

    use_columns = ["permno", "date", "ret", "price", "market_cap", "volume"]
    trailing = pd.DataFrame(columns=use_columns)
    weekly_parts: list[pd.DataFrame] = []
    for path in files:
        file_range = _file_range(path)
        if file_range is not None:
            file_start, file_end = file_range
            if start is not None and file_end < start - pd.Timedelta(days=45):
                continue
            if end is not None and file_start > end:
                continue
        raw = pd.read_csv(
            path,
            usecols=use_columns,
            dtype={"permno": "string"},
            parse_dates=["date"],
        )
        if start is not None:
            raw = raw.loc[raw["date"] >= start - pd.Timedelta(days=45)]
        if end is not None:
            raw = raw.loc[raw["date"] <= end]
        if raw.empty:
            continue
        current_start = raw["date"].min()
        panel = pd.concat([trailing, raw], ignore_index=True)
        panel["permno"] = panel["permno"].astype(str)
        panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
        panel = panel.loc[panel["date"].notna()]
        panel = panel.drop_duplicates(["permno", "date"], keep="last")
        panel = panel.sort_values(["permno", "date"])
        numeric = ["ret", "price", "market_cap", "volume"]
        panel[numeric] = panel[numeric].apply(pd.to_numeric, errors="coerce")
        panel["price"] = panel["price"].abs()
        panel["market_cap_usd"] = panel["market_cap"].abs() * market_cap_scale
        panel["dollar_volume"] = panel["price"] * panel["volume"].clip(lower=0.0)
        panel["turnover"] = panel["dollar_volume"] / panel["market_cap_usd"].replace(
            0.0, np.nan
        )
        panel["amihud_daily"] = (
            panel["ret"].abs() / panel["dollar_volume"].replace(0.0, np.nan) * 1e6
        )

        grouped = panel.groupby("permno", sort=False)
        panel["dollar_volume_20d"] = grouped["dollar_volume"].transform(
            lambda values: values.rolling(rolling_days, min_periods=5).median()
        )
        panel["turnover_20d"] = grouped["turnover"].transform(
            lambda values: values.rolling(rolling_days, min_periods=5).mean()
        )
        panel["amihud_20d"] = grouped["amihud_daily"].transform(
            lambda values: values.rolling(rolling_days, min_periods=5).mean()
        )
        panel["realized_volatility_20d"] = grouped["ret"].transform(
            lambda values: values.rolling(rolling_days, min_periods=5).std(ddof=1)
        )
        panel["zero_return_fraction_20d"] = grouped["ret"].transform(
            lambda values: values.eq(0.0).rolling(rolling_days, min_periods=5).mean()
        )

        # Signals formed at Friday close are deliberately shifted one weekly
        # period before export.  A target formed on date t therefore consumes
        # only market/microstructure information through the prior Friday.
        panel["formation_date"] = panel["date"].dt.to_period("W-FRI").dt.end_time.dt.normalize()
        weekly = (
            panel.sort_values(["permno", "date"])
            .groupby(["formation_date", "permno"], as_index=False)
            .tail(1)
        )
        feature_columns = [
            "market_cap_usd",
            "price",
            "dollar_volume_20d",
            "turnover_20d",
            "amihud_20d",
            "realized_volatility_20d",
            "zero_return_fraction_20d",
        ]
        weekly[feature_columns] = weekly.groupby("permno")[feature_columns].shift(1)
        current_week = current_start.to_period("W-FRI").end_time.normalize()
        weekly = weekly.loc[weekly["formation_date"] >= current_week]
        weekly = weekly.rename(columns={"permno": "asset"})
        weekly_parts.append(weekly[["formation_date", "asset", *feature_columns]])

        trailing = (
            panel.groupby("permno", group_keys=False)
            .tail(rolling_days + 2)[use_columns]
            .copy()
        )

    if not weekly_parts:
        raise RuntimeError("No structural feature rows were produced.")
    output = pd.concat(weekly_parts, ignore_index=True)
    output = output.drop_duplicates(["formation_date", "asset"], keep="last")
    if start is not None:
        output = output.loc[output["formation_date"] >= start]
    if end is not None:
        output = output.loc[output["formation_date"] <= end]
    return output.sort_values(["formation_date", "asset"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    features = build_structural_features(
        args.daily_cache_dir,
        start=args.start,
        end=args.end,
        rolling_days=args.rolling_days,
        market_cap_scale=args.market_cap_scale,
    )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(destination, index=False)
    print(f"rows: {len(features):,}")
    print(f"assets: {features['asset'].nunique():,}")
    print(f"dates: {features['formation_date'].min().date()} through {features['formation_date'].max().date()}")
    print(f"saved: {destination.resolve()}")


if __name__ == "__main__":
    main()
