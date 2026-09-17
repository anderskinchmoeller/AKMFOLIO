"""Build weekly closing-auction / order-flow features from CRSP daily OHLC.

Two steps (run from the project root)::

    # 1. Download CRSP CIZ daily bars incl. open/high/low/close/bid/ask into a
    #    separate, resumable cache (your existing daily_cache lacks OHLC).
    python -m akm_hrp.cli.build_microstructure_features --download \
        --start 1989-01-01 --end 2025-12-31 \
        --cache-dir wrds_data/daily_ohlc_cache

    # 2. (Re)build the weekly feature file from that cache without WRDS.
    python -m akm_hrp.cli.build_microstructure_features \
        --cache-dir wrds_data/daily_ohlc_cache \
        --universe data/weekly_returns.csv \
        --output data/microstructure_alpha_features.csv.gz

``compare_models`` picks up ``<bundle>/microstructure_alpha_features.csv.gz``
automatically when it exists.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from akm_hrp.data.microstructure_features import (
    MICROSTRUCTURE_FEATURE_COLUMNS,
    OHLC_COLUMNS,
    REQUIRED_TRAILING_DAYS,
    weekly_microstructure_features,
)

_CACHE_DATE_PATTERN = re.compile(r"_(\d{8})_(\d{8})_")
_DAILY_COLUMNS = (
    "permno", "date", "ret", "price", "volume",
    "open", "high", "low", "close", "bid", "ask", "num_trades",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--download", action="store_true",
                        help="Pull CRSP CIZ daily OHLC chunks from WRDS first.")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--library", default=None)
    parser.add_argument("--wrds-username", default=None)
    parser.add_argument("--chunk-months", type=int, default=12)
    parser.add_argument("--cache-dir", default="wrds_data/daily_ohlc_cache")
    parser.add_argument(
        "--universe",
        default=None,
        help="Optional weekly_returns.csv; only its permno columns are kept.",
    )
    parser.add_argument("--output", default="data/microstructure_alpha_features.csv.gz")
    parser.add_argument(
        "--lag-weeks",
        type=int,
        default=0,
        help=(
            "Extra weekly lag. 0 (default) = data through the formation Friday "
            "close, trades assumed next session. 1 = the conservative "
            "convention of build_structural_alpha_features."
        ),
    )
    return parser.parse_args()


def download(args: argparse.Namespace) -> None:
    try:
        import wrds
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Install the 'wrds' package to use --download.") from exc
    from akm_hrp.data.wrds_crsp import (
        CRSPChunkConfig,
        CRSPQueryConfig,
        iter_crsp_ciz_daily_chunks,
        resolve_crsp_table,
    )

    if not args.start or not args.end:
        raise ValueError("--download requires --start and --end.")
    connection = wrds.Connection(wrds_username=args.wrds_username)
    try:
        query = CRSPQueryConfig(library=args.library)
        library, table, columns = resolve_crsp_table(connection, query)
        missing = [c for c in OHLC_COLUMNS if c not in columns]
        if missing:
            raise RuntimeError(
                f"{library}.{table} exposes no {missing} fields; closing-auction "
                "features need CIZ DlyOpen/DlyHigh/DlyLow/DlyClose."
            )
        chunks = iter_crsp_ciz_daily_chunks(
            connection,
            args.start,
            args.end,
            query_config=query,
            chunk_config=CRSPChunkConfig(
                months_per_chunk=args.chunk_months,
                cache_dir=args.cache_dir,
                resume=True,
            ),
        )
        for chunk in chunks:
            print(
                f"chunk {chunk.attrs['chunk_number']}/{chunk.attrs['chunk_count']} "
                f"{chunk.attrs['chunk_start']}..{chunk.attrs['chunk_end']} "
                f"rows={len(chunk):,} cached={chunk.attrs['chunk_cache_hit']}",
                flush=True,
            )
    finally:
        connection.close()


def _file_start(path: Path) -> str:
    match = _CACHE_DATE_PATTERN.search(path.name)
    return match.group(1) if match else path.name


def build(
    cache_dir: str | Path,
    *,
    universe: set[str] | None = None,
    lag_weeks: int = 0,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    files = sorted(Path(cache_dir).glob("*.csv.gz"), key=_file_start)
    if not files:
        raise FileNotFoundError(f"No .csv.gz chunks in {cache_dir}")
    trailing = pd.DataFrame()
    parts: list[pd.DataFrame] = []
    for path in files:
        header = pd.read_csv(path, nrows=0).columns
        missing = [c for c in OHLC_COLUMNS if c not in header]
        if missing:
            raise ValueError(
                f"{path.name} has no {missing}; it was downloaded without OHLC. "
                "Use a fresh --cache-dir with --download."
            )
        raw = pd.read_csv(
            path,
            usecols=[c for c in _DAILY_COLUMNS if c in header],
            dtype={"permno": "string"},
            parse_dates=["date"],
        )
        if universe is not None:
            raw = raw.loc[raw["permno"].isin(universe)]
        if end is not None:
            raw = raw.loc[raw["date"] <= end]
        if raw.empty:
            continue
        first_new_week = raw["date"].min().to_period("W-FRI").end_time.normalize()
        panel = pd.concat([trailing, raw], ignore_index=True)
        weekly = weekly_microstructure_features(panel, lag_weeks=0)
        weekly = weekly.loc[weekly["formation_date"] >= first_new_week]
        parts.append(weekly)
        panel = panel.sort_values(["permno", "date"])
        trailing = panel.groupby("permno", group_keys=False).tail(
            REQUIRED_TRAILING_DAYS + 5 * max(lag_weeks, 0)
        )
        print(f"{path.name}: weekly rows {len(weekly):,}", flush=True)
    if not parts:
        raise RuntimeError("No microstructure rows were produced.")
    out = pd.concat(parts, ignore_index=True)
    out = out.drop_duplicates(["formation_date", "asset"], keep="last")
    out = out.sort_values(["asset", "formation_date"])
    if lag_weeks:
        cols = list(MICROSTRUCTURE_FEATURE_COLUMNS)
        out[cols] = out.groupby("asset")[cols].shift(int(lag_weeks))
    if start is not None:
        out = out.loc[out["formation_date"] >= start]
    return out.sort_values(["formation_date", "asset"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    if args.download:
        download(args)
    universe = None
    if args.universe:
        columns = pd.read_csv(args.universe, nrows=0).columns
        universe = {str(c) for c in columns if str(c) != "date"}
    features = build(
        args.cache_dir,
        universe=universe,
        lag_weeks=args.lag_weeks,
        start=pd.Timestamp(args.start) if args.start else None,
        end=pd.Timestamp(args.end) if args.end else None,
    )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(destination, index=False, float_format="%.6g")
    print(f"rows: {len(features):,}  assets: {features['asset'].nunique():,}")
    print(f"saved: {destination.resolve()}")


if __name__ == "__main__":
    main()
