from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from akm_hrp.data.hrp_universe import (
    CRSP_HRP_ETF_TEMPLATE,
    UniverseAsset,
    prepare_hrp_distance_inputs,
)
from akm_hrp.data.wrds_crsp import (
    CRSPChunkConfig,
    CRSPQueryConfig,
    CRSPWeeklyBundle,
    CRSPWeeklyConfig,
    build_crsp_weekly_bundle_from_chunks,
    iter_crsp_ciz_daily_chunks,
    save_crsp_weekly_bundle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a liquid, cross-exposure ETF/ETV universe from CRSP CIZ and "
            "export HRP-ready rank-correlation distance inputs."
        )
    )
    parser.add_argument("--start", required=True, help="First daily date (YYYY-MM-DD).")
    parser.add_argument("--end", required=True, help="Last daily date (YYYY-MM-DD).")
    parser.add_argument("--output-dir", default="wrds_hrp_etf")
    parser.add_argument("--library", default=None)
    parser.add_argument("--wrds-username", default=None)

    universe = parser.add_mutually_exclusive_group()
    universe.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        help="Custom ETF/ETV ticker list; defaults to the documented 15-asset template.",
    )
    universe.add_argument(
        "--universe-file",
        default=None,
        help="CSV with ticker and optional cluster/role columns.",
    )

    parser.add_argument("--minimum-history-days", type=int, default=252)
    parser.add_argument("--minimum-price", type=float, default=5.0)
    parser.add_argument(
        "--minimum-median-dollar-volume",
        type=float,
        default=1_000_000.0,
        help="63-day median price-times-volume floor (default: 1,000,000).",
    )
    parser.add_argument("--liquidity-window-days", type=int, default=63)
    parser.add_argument(
        "--minimum-complete-weeks",
        type=int,
        default=104,
        help="Minimum common PIT-eligible history for distance estimation.",
    )
    parser.add_argument(
        "--security-subtypes",
        nargs="+",
        default=["ETF", "ETV"],
        help="Accepted CRSP CIZ fund subtypes (default: ETF ETV).",
    )
    parser.add_argument("--chunk-months", type=int, default=12)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse completed filtered CRSP chunks (default: enabled).",
    )
    return parser.parse_args()


def _load_universe(args: argparse.Namespace) -> tuple[UniverseAsset, ...]:
    if args.universe_file:
        source = pd.read_csv(args.universe_file, dtype=str).fillna("")
        if "ticker" not in source.columns:
            raise ValueError("--universe-file must contain a ticker column.")
        assets = tuple(
            UniverseAsset(
                ticker=str(row.ticker).strip().upper(),
                cluster=(str(getattr(row, "cluster", "")).strip() or "custom"),
                role=(str(getattr(row, "role", "")).strip() or "custom exposure"),
            )
            for row in source.itertuples(index=False)
        )
    elif args.tickers:
        assets = tuple(
            UniverseAsset(str(ticker).strip().upper(), "custom", "custom exposure")
            for ticker in args.tickers
        )
    else:
        assets = CRSP_HRP_ETF_TEMPLATE

    tickers = [asset.ticker for asset in assets]
    if not tickers or any(not ticker for ticker in tickers):
        raise ValueError("The HRP universe must contain at least one non-empty ticker.")
    duplicates = pd.Index(tickers)[pd.Index(tickers).duplicated()].unique().tolist()
    if duplicates:
        raise ValueError(f"Duplicate requested tickers: {duplicates}")
    if len(tickers) < 2:
        raise ValueError("The HRP universe must contain at least two assets.")
    return assets


def _resolve_requested_assets(
    bundle: CRSPWeeklyBundle,
    universe: tuple[UniverseAsset, ...],
) -> tuple[CRSPWeeklyBundle, pd.DataFrame]:
    metadata = bundle.metadata.copy()
    metadata["ticker"] = metadata["ticker"].astype(str).str.strip().str.upper()
    metadata["asset"] = metadata["permno"].astype(str)

    ambiguity = metadata.groupby("ticker")["asset"].nunique().gt(1)
    requested = {asset.ticker for asset in universe}
    ambiguous_requested = sorted(set(ambiguity[ambiguity].index) & requested)
    if ambiguous_requested:
        raise ValueError(
            "Requested ticker maps to multiple CRSP PERMNOs in the selected "
            f"period: {ambiguous_requested}. Use a PERMNO-specific universe."
        )

    ticker_to_asset = (
        metadata.drop_duplicates("ticker", keep="last").set_index("ticker")["asset"]
    )
    missing = [asset.ticker for asset in universe if asset.ticker not in ticker_to_asset]
    if missing:
        raise ValueError(
            "CRSP did not resolve the requested ETF/ETV tickers under the "
            f"point-in-time classification filter: {missing}"
        )

    asset_order = [ticker_to_asset.loc[asset.ticker] for asset in universe]
    if len(set(asset_order)) != len(asset_order):
        raise ValueError("Multiple requested tickers resolve to the same CRSP PERMNO.")

    definitions = pd.DataFrame([asdict(asset) for asset in universe])
    universe_table = definitions.merge(
        metadata.loc[:, ["asset", "permno", "ticker"]],
        on="ticker",
        how="left",
        validate="one_to_one",
    )
    universe_table = universe_table.loc[:, ["asset", "permno", "ticker", "cluster", "role"]]

    ordered_metadata = metadata.set_index("asset").loc[asset_order].reset_index()
    ordered_metadata = ordered_metadata.merge(
        definitions,
        on="ticker",
        how="left",
        validate="one_to_one",
    )
    diagnostics = dict(bundle.diagnostics)
    diagnostics.update(
        {
            "universe_method": "explicit_point_in_time_crsp_etf_etv_template",
            "requested_tickers": [asset.ticker for asset in universe],
            "resolved_assets": int(len(asset_order)),
        }
    )
    ordered = CRSPWeeklyBundle(
        returns=bundle.returns.reindex(columns=asset_order),
        pit_universe=bundle.pit_universe.reindex(columns=asset_order).fillna(0).astype(int),
        metadata=ordered_metadata.drop(columns=["asset"]),
        diagnostics=diagnostics,
    )
    return ordered, universe_table


def main() -> None:
    args = parse_args()
    universe = _load_universe(args)
    output_dir = Path(args.output_dir)
    cache_dir = args.cache_dir or str(output_dir / "daily_cache")

    try:
        import wrds
    except ImportError as exc:  # pragma: no cover - optional remote dependency
        raise RuntimeError("Install the 'wrds' package before using this command.") from exc

    connection = wrds.Connection(wrds_username=args.wrds_username)
    try:
        chunks = iter_crsp_ciz_daily_chunks(
            connection,
            args.start,
            args.end,
            query_config=CRSPQueryConfig(
                library=args.library,
                common_stocks_only=False,
                tickers=tuple(asset.ticker for asset in universe),
                share_types=("NS",),
                security_types=("FUND",),
                security_subtypes=tuple(
                    str(value).strip().upper() for value in args.security_subtypes
                ),
            ),
            chunk_config=CRSPChunkConfig(
                months_per_chunk=args.chunk_months,
                cache_dir=cache_dir,
                resume=args.resume,
            ),
        )
        raw_bundle = build_crsp_weekly_bundle_from_chunks(
            chunks,
            CRSPWeeklyConfig(
                minimum_history_days=args.minimum_history_days,
                liquidity_window_days=args.liquidity_window_days,
                minimum_price=args.minimum_price,
                minimum_median_dollar_volume=args.minimum_median_dollar_volume,
                top_n_by_market_cap=None,
            ),
        )
    finally:
        connection.close()

    bundle, universe_table = _resolve_requested_assets(raw_bundle, universe)
    distance_inputs = prepare_hrp_distance_inputs(
        bundle.returns,
        pit_universe=bundle.pit_universe,
        minimum_complete_weeks=args.minimum_complete_weeks,
    )
    bundle.diagnostics.update(
        {
            "distance_estimator": "Ledoit-Wolf shrinkage of standardized ranks",
            "distance_metric": "angular distance from denoised rank correlation",
            "distance_sample_weeks": int(len(distance_inputs.returns)),
            "distance_shrinkage_intensity": distance_inputs.shrinkage_intensity,
            "minimum_complete_weeks": int(args.minimum_complete_weeks),
        }
    )

    save_crsp_weekly_bundle(bundle, output_dir)
    universe_table.to_csv(output_dir / "hrp_universe.csv", index=False)
    distance_inputs.returns.to_csv(output_dir / "hrp_complete_returns.csv")
    distance_inputs.volatility_normalized_returns.to_csv(
        output_dir / "hrp_volatility_normalized_returns.csv"
    )
    distance_inputs.spearman_correlation.to_csv(
        output_dir / "hrp_spearman_correlation.csv"
    )
    distance_inputs.denoised_correlation.to_csv(
        output_dir / "hrp_denoised_correlation.csv"
    )
    distance_inputs.angular_distance.to_csv(
        output_dir / "hrp_angular_distance.csv"
    )

    print(universe_table.to_string(index=False))
    print(f"complete HRP weeks: {len(distance_inputs.returns)}")
    print(f"rank-correlation shrinkage: {distance_inputs.shrinkage_intensity:.6f}")
    print(f"saved: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
