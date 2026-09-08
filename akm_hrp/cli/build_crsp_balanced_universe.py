from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from akm_hrp.data.balanced_universe import (
    BalancedUniverseConfig,
    attach_point_in_time_sectors,
    build_style_factor_proxies,
    merge_point_in_time_characteristics,
    monthly_membership_to_weekly_pit,
    select_balanced_monthly_membership,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a sector-balanced, lagged CRSP common-stock HRP universe "
            "with optional weekly CRSP Treasury/index sleeves."
        )
    )
    parser.add_argument(
        "--returns",
        default="wrds_full_clean/weekly_returns.csv",
    )
    parser.add_argument(
        "--base-pit",
        default="wrds_full_clean/pit_universe_delisting_safe.csv",
        help="Broad safety/eligibility mask intersected after balanced selection.",
    )
    parser.add_argument(
        "--structural-features",
        default="wrds_full_clean/structural_alpha_features.csv.gz",
        help="Lagged CRSP market-cap/liquidity feature table.",
    )
    parser.add_argument(
        "--sector-history",
        default="wrds_full_clean/crsp_sector_history.csv.gz",
        help="Date-effective CRSP UES/ICB/SIC history.",
    )
    parser.add_argument(
        "--value-features",
        default=None,
        help=(
            "Optional release-aware weekly Compustat feature file containing "
            "formation_date, PERMNO/asset, and book_to_market."
        ),
    )
    parser.add_argument(
        "--asset-metadata",
        default="wrds_full_clean/crsp_security_metadata.csv",
    )
    parser.add_argument(
        "--macro-returns",
        default=None,
        help=(
            "Optional wide weekly decimal-return CSV from CRSP Treasury/index "
            "series, for example 1Y/5Y/10Y/30Y sleeves."
        ),
    )
    parser.add_argument("--output-dir", default="wrds_balanced_hrp")
    parser.add_argument("--stocks-per-sector", type=int, default=5)
    parser.add_argument("--minimum-size-percentile", type=float, default=0.80)
    parser.add_argument("--minimum-price", type=float, default=5.0)
    parser.add_argument(
        "--minimum-median-dollar-volume",
        type=float,
        default=1_000_000.0,
    )
    parser.add_argument("--minimum-sectors", type=int, default=6)
    parser.add_argument("--selection-lag-months", type=int, default=1)
    parser.add_argument("--minimum-macro-history-weeks", type=int, default=52)
    parser.add_argument("--factor-minimum-leg-assets", type=int, default=10)
    parser.add_argument("--factor-quantile", type=float, default=0.30)
    parser.add_argument(
        "--factor-minimum-realized-leg-fraction",
        type=float,
        default=0.80,
    )
    parser.add_argument(
        "--include-factor-proxies",
        action="store_true",
        help=(
            "Append zero-investment size/value/momentum/low-vol proxies to the "
            "investable matrix. Omit unless both long and short legs are tradable."
        ),
    )
    return parser.parse_args()


def _load_wide(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError(f"{path} must use parseable dates in its first column.")
    if frame.empty:
        raise ValueError(f"{path} contains no rows.")
    frame.columns = frame.columns.astype(str)
    if frame.index.has_duplicates:
        raise ValueError(f"{path} contains duplicate dates.")
    return frame.apply(pd.to_numeric, errors="coerce")


def _load_weekly_sleeves(path: str | Path) -> pd.DataFrame:
    """Load one return per week and normalize holiday labels to Friday."""

    frame = _load_wide(path)
    week_end = frame.index.to_period("W-FRI").end_time.normalize()
    if week_end.duplicated().any():
        raise ValueError(
            f"{path} has multiple observations in a week; compound daily returns first."
        )
    frame.index = week_end
    return frame.sort_index()


def _complete_sleeve_pit(
    returns: pd.DataFrame,
    minimum_history_weeks: int,
) -> pd.DataFrame:
    """Activate a sleeve after a contiguous minimum return history."""

    if minimum_history_weeks < 2:
        raise ValueError("minimum_history_weeks must be at least 2.")
    valid = returns.notna()
    interior_gap = (~valid) & valid.cummax() & valid.iloc[::-1].cummax().iloc[::-1]
    if interior_gap.any().any():
        affected = interior_gap.any(axis=0)
        raise ValueError(
            "Macro/factor sleeve returns contain interior gaps: "
            f"{affected[affected].index.tolist()}"
        )
    return (valid & (valid.cumsum() >= int(minimum_history_weeks))).astype(np.int8)


def _build_output_metadata(
    asset_metadata_path: str | Path,
    stock_assets: pd.Index,
    membership: pd.DataFrame,
    factor_assets: list[str],
    macro_assets: list[str],
) -> pd.DataFrame:
    """Create ticker labels and economic cluster labels for every output asset."""

    metadata = pd.read_csv(asset_metadata_path, dtype=str)
    if "permno" not in metadata:
        raise ValueError("Asset metadata must contain a permno column.")
    metadata["permno"] = (
        metadata["permno"]
        .astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )
    metadata = metadata.drop_duplicates("permno", keep="last")
    if "ticker" not in metadata:
        metadata["ticker"] = metadata["permno"]

    latest_sector = (
        membership.sort_values("formation_date")
        .drop_duplicates("asset", keep="last")
        .set_index("asset")["sector"]
    )
    stocks = pd.DataFrame({"permno": stock_assets.astype(str)}).merge(
        metadata,
        on="permno",
        how="left",
        validate="one_to_one",
    )
    stocks["asset"] = stocks["permno"]
    stocks["ticker"] = stocks["ticker"].fillna(stocks["asset"])
    stocks["cluster"] = "equity:" + stocks["asset"].map(latest_sector).fillna(
        "unknown"
    ).astype(str)
    stocks["role"] = "sector-balanced liquid common stock"
    columns = ["asset", "permno", "ticker", "cluster", "role"]

    sleeves = [
        {
            "asset": asset,
            "permno": "",
            "ticker": asset,
            "cluster": "factor:style",
            "role": "zero-investment CRSP style proxy",
        }
        for asset in factor_assets
    ]
    sleeves.extend(
        {
            "asset": asset,
            "permno": "",
            "ticker": asset,
            "cluster": "macro:fixed_income_or_index",
            "role": "CRSP Treasury or index sleeve",
        }
        for asset in macro_assets
    )
    if not sleeves:
        return stocks.loc[:, columns]
    return pd.concat(
        [stocks.loc[:, columns], pd.DataFrame(sleeves)],
        ignore_index=True,
    )


def _write_outputs(
    destination: Path,
    returns: pd.DataFrame,
    pit: pd.DataFrame,
    membership: pd.DataFrame,
    factor_returns: pd.DataFrame,
    metadata: pd.DataFrame,
    manifest: dict[str, object],
) -> None:
    """Write one internally consistent universe bundle."""

    destination.mkdir(parents=True, exist_ok=True)
    returns.to_csv(destination / "weekly_returns.csv")
    pit.to_csv(destination / "pit_universe.csv")
    membership.to_csv(destination / "selection_history.csv", index=False)
    factor_returns.to_csv(destination / "crsp_style_factor_returns.csv")
    metadata.to_csv(destination / "crsp_security_metadata.csv", index=False)
    (destination / "universe_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    returns = _load_wide(args.returns)
    base_pit = _load_wide(args.base_pit).fillna(0).astype(bool)
    features = pd.read_csv(args.structural_features, low_memory=False)
    if args.value_features:
        value_features = pd.read_csv(args.value_features, low_memory=False)
        features = merge_point_in_time_characteristics(
            features,
            value_features,
            columns=("book_to_market",),
        )
    sector_history = pd.read_csv(args.sector_history, low_memory=False)

    point_in_time = attach_point_in_time_sectors(features, sector_history)
    # Select only assets that have a corresponding return column.  Previously
    # this stage ranked the entire CRSP feature panel and only intersected with
    # the supplied return matrix afterwards.  That silently discarded most
    # historic selections and could leave the output investable only in recent
    # years even when the return matrix itself had a long history.
    return_assets = pd.Index(returns.columns.astype(str))
    investable_features = point_in_time.loc[
        point_in_time["asset"].isin(return_assets)
    ].copy()
    if investable_features.empty:
        raise RuntimeError(
            "No structural-feature assets match the supplied return columns."
        )
    config = BalancedUniverseConfig(
        stocks_per_sector=args.stocks_per_sector,
        minimum_size_percentile=args.minimum_size_percentile,
        minimum_price=args.minimum_price,
        minimum_median_dollar_volume=args.minimum_median_dollar_volume,
        minimum_sectors=args.minimum_sectors,
        selection_lag_months=args.selection_lag_months,
    )
    membership = select_balanced_monthly_membership(investable_features, config)
    equity_pit = monthly_membership_to_weekly_pit(
        membership,
        returns,
        base_pit=base_pit,
    )
    combined_returns = returns.reindex(columns=equity_pit.columns)
    combined_pit = equity_pit.copy()

    factor_returns = build_style_factor_proxies(
        returns,
        point_in_time,
        eligible_mask=base_pit,
        minimum_leg_assets=args.factor_minimum_leg_assets,
        quantile=args.factor_quantile,
        minimum_price=args.minimum_price,
        minimum_median_dollar_volume=args.minimum_median_dollar_volume,
        minimum_realized_leg_fraction=args.factor_minimum_realized_leg_fraction,
    ).dropna(axis=1, how="all")
    factor_assets: list[str] = []
    if args.include_factor_proxies:
        # A long-short factor is not a cash asset unless both legs are
        # implementable; the opt-in flag makes that assumption explicit.
        factor_pit = _complete_sleeve_pit(
            factor_returns,
            args.minimum_macro_history_weeks,
        )
        combined_returns = pd.concat([combined_returns, factor_returns], axis=1)
        combined_pit = pd.concat([combined_pit, factor_pit], axis=1)
        factor_assets = factor_returns.columns.tolist()

    macro_columns: list[str] = []
    if args.macro_returns:
        macro = _load_weekly_sleeves(args.macro_returns).reindex(returns.index)
        overlap = combined_returns.columns.intersection(macro.columns)
        if len(overlap):
            raise ValueError(
                f"Macro return names collide with CRSP assets: {list(overlap)}"
            )
        macro_pit = _complete_sleeve_pit(
            macro,
            args.minimum_macro_history_weeks,
        )
        combined_returns = pd.concat([combined_returns, macro], axis=1)
        combined_pit = pd.concat([combined_pit, macro_pit], axis=1)
        macro_columns = macro.columns.tolist()

    active_columns = combined_pit.columns[combined_pit.sum(axis=0) > 0]
    combined_returns = combined_returns.reindex(columns=active_columns)
    combined_pit = combined_pit.reindex(columns=active_columns).astype(np.int8)
    if combined_returns.shape[1] < 2:
        raise RuntimeError("Balanced CRSP universe contains fewer than two assets.")

    selected_stock_columns = equity_pit.columns.intersection(active_columns)
    output_metadata = _build_output_metadata(
        args.asset_metadata,
        selected_stock_columns,
        membership,
        factor_assets,
        macro_columns,
    )
    manifest = {
        "method": "lagged_sector_balanced_crsp_common_stocks_plus_optional_macro",
        "n_weeks": len(combined_returns),
        "n_assets": int(combined_returns.shape[1]),
        "first_week": combined_returns.index.min().date().isoformat(),
        "last_week": combined_returns.index.max().date().isoformat(),
        "maximum_weekly_eligible_assets": int(combined_pit.sum(axis=1).max()),
        "median_weekly_eligible_assets": float(combined_pit.sum(axis=1).median()),
        "first_week_with_two_eligible_assets": (
            combined_pit.index[combined_pit.sum(axis=1) >= 2].min().date().isoformat()
        ),
        "macro_columns": macro_columns,
        "factor_columns": factor_returns.columns.tolist(),
        "value_features_supplied": bool(args.value_features),
        "factor_proxies_in_investable_universe": bool(args.include_factor_proxies),
        "config": asdict(config),
        "factor_config": {
            "minimum_leg_assets": int(args.factor_minimum_leg_assets),
            "quantile": float(args.factor_quantile),
            "minimum_realized_leg_fraction": float(
                args.factor_minimum_realized_leg_fraction
            ),
        },
    }
    destination = Path(args.output_dir)
    _write_outputs(
        destination,
        combined_returns,
        combined_pit,
        membership,
        factor_returns,
        output_metadata,
        manifest,
    )

    print(f"assets: {combined_returns.shape[1]:,}")
    print(f"maximum weekly eligible: {combined_pit.sum(axis=1).max():,}")
    print(f"median weekly eligible: {combined_pit.sum(axis=1).median():,.1f}")
    first_tradeable = combined_pit.index[combined_pit.sum(axis=1) >= 2].min()
    print(f"first week with at least two eligible assets: {first_tradeable.date()}")
    print(f"saved: {destination.resolve()}")


if __name__ == "__main__":
    main()
