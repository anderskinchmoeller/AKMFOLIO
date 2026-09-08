from __future__ import annotations

import argparse
from pathlib import Path

from akm_hrp.data.wrds_crsp import (
    CRSPChunkConfig,
    CRSPQueryConfig,
    CRSPWeeklyConfig,
    build_crsp_weekly_bundle_from_chunks,
    iter_crsp_ciz_daily_chunks,
    save_crsp_weekly_bundle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build PIT weekly HRP inputs from WRDS CRSP CIZ."
    )
    parser.add_argument("--start", required=True, help="First daily date (YYYY-MM-DD).")
    parser.add_argument("--end", required=True, help="Last daily date (YYYY-MM-DD).")
    parser.add_argument("--output-dir", default="wrds_data")
    parser.add_argument(
        "--library",
        default=None,
        help="WRDS schema override. Default: auto-discover crsp, then crsp_a_stock.",
    )
    parser.add_argument("--wrds-username", default=None)
    parser.add_argument("--minimum-history-days", type=int, default=252)
    parser.add_argument("--minimum-price", type=float, default=5.0)
    parser.add_argument("--minimum-market-cap", type=float, default=None)
    parser.add_argument("--maximum-market-cap", type=float, default=None)
    parser.add_argument("--minimum-median-dollar-volume", type=float, default=None)
    parser.add_argument(
        "--maximum-beta",
        type=float,
        default=None,
        help="Optional maximum lagged weekly market beta, e.g. 0.80.",
    )
    parser.add_argument(
        "--beta-lookback-weeks",
        type=int,
        default=52,
        help="Rolling beta estimation window (default: 52 weeks).",
    )
    parser.add_argument(
        "--beta-minimum-observations",
        type=int,
        default=26,
        help="Minimum lagged observations required for beta (default: 26).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=1500,
        help=(
            "Maximum stocks selected from lagged monthly market-cap ranks. "
            "Use 0 to disable ranking. Default: 1500."
        ),
    )
    parser.add_argument(
        "--ranking-lag-months",
        type=int,
        default=1,
        help="Months between market-cap formation and eligibility. Default: 1.",
    )
    parser.add_argument(
        "--chunk-months",
        type=int,
        default=12,
        help="Months requested per restartable WRDS chunk. Default: 12.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Daily chunk cache. Default: <output-dir>/daily_cache.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse completed daily chunks (default: enabled).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import wrds
    except ImportError as exc:  # pragma: no cover - optional remote dependency
        raise RuntimeError("Install the 'wrds' package before using this command.") from exc

    if args.top_n < 0:
        raise ValueError("--top-n must be zero or a positive integer.")
    cache_dir = args.cache_dir or str(Path(args.output_dir) / "daily_cache")

    connection = wrds.Connection(wrds_username=args.wrds_username)
    try:
        chunks = iter_crsp_ciz_daily_chunks(
            connection,
            args.start,
            args.end,
            query_config=CRSPQueryConfig(library=args.library),
            chunk_config=CRSPChunkConfig(
                months_per_chunk=args.chunk_months,
                cache_dir=cache_dir,
                resume=args.resume,
            ),
        )
        bundle = build_crsp_weekly_bundle_from_chunks(
            chunks,
            CRSPWeeklyConfig(
                minimum_history_days=args.minimum_history_days,
                minimum_price=args.minimum_price,
                minimum_market_cap=args.minimum_market_cap,
                maximum_market_cap=args.maximum_market_cap,
                minimum_median_dollar_volume=args.minimum_median_dollar_volume,
                maximum_beta=args.maximum_beta,
                beta_lookback_weeks=args.beta_lookback_weeks,
                beta_minimum_observations=args.beta_minimum_observations,
                top_n_by_market_cap=args.top_n or None,
                ranking_lag_months=args.ranking_lag_months,
            ),
        )
    finally:
        connection.close()
    save_crsp_weekly_bundle(bundle, args.output_dir)
    for key, value in bundle.diagnostics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
