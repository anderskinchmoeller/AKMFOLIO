from __future__ import annotations

import argparse
import json
from pathlib import Path

from akm_hrp.data.wrds_treasury import (
    CRSP_FIXED_TERM_SERIES,
    CRSPTreasuryQueryConfig,
    build_crsp_treasury_weekly,
    fetch_crsp_fixed_term_daily,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and compound CRSP fixed-term Treasury index returns."
    )
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument(
        "--series",
        type=int,
        nargs="+",
        default=list(CRSP_FIXED_TERM_SERIES),
        help="CRSP TREASNOX identifiers (default: 1Y, 5Y, 10Y, and 30Y).",
    )
    parser.add_argument(
        "--output",
        default="wrds_full_clean/crsp_treasury_weekly_returns.csv",
    )
    parser.add_argument("--library", default=None)
    parser.add_argument("--wrds-username", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    unknown = sorted(set(args.series).difference(CRSP_FIXED_TERM_SERIES))
    if unknown:
        raise ValueError(
            f"Unknown fixed-term TREASNOX values: {unknown}. "
            f"Supported: {sorted(CRSP_FIXED_TERM_SERIES)}"
        )
    series = {
        identifier: CRSP_FIXED_TERM_SERIES[identifier] for identifier in args.series
    }
    try:
        import wrds
    except ImportError as exc:  # pragma: no cover - optional remote dependency
        raise RuntimeError(
            "Install the 'wrds' package before using this command."
        ) from exc

    connection = wrds.Connection(wrds_username=args.wrds_username)
    try:
        daily = fetch_crsp_fixed_term_daily(
            connection,
            args.start,
            args.end,
            series=series,
            config=CRSPTreasuryQueryConfig(library=args.library),
        )
    finally:
        connection.close()
    weekly = build_crsp_treasury_weekly(daily)

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    weekly.to_csv(destination)
    manifest = {
        "source": "WRDS CRSP US Treasury daily fixed-term indexes",
        "wrds_library": daily.attrs.get("wrds_library"),
        "wrds_table": daily.attrs.get("wrds_table"),
        "return_field": daily.attrs.get("return_field"),
        "series": {str(key): value for key, value in series.items()},
        "first_week": weekly.index.min().date().isoformat(),
        "last_week": weekly.index.max().date().isoformat(),
        "n_weeks": len(weekly),
    }
    destination.with_suffix(destination.suffix + ".metadata.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"weeks: {len(weekly):,}")
    print(f"series: {', '.join(weekly.columns)}")
    print(f"saved: {destination.resolve()}")


if __name__ == "__main__":
    main()
