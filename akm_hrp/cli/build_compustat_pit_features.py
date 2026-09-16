#!/usr/bin/env python3
"""Build release-date-aware Compustat features for a weekly CRSP universe.

Inputs
------
* Long quarterly Compustat/CCM extract produced from ``comp.fundq``.
* Wide weekly CRSP return CSV (date index, PERMNO columns).
* Matching point-in-time universe mask.

Output
------
A sparse long feature file keyed by ``formation_date, permno``.  Every source
record satisfies ``available_date <= formation_date`` and is discarded after a
configurable staleness limit.  The output intentionally contains raw causal
features; winsorization and cross-sectional standardization must occur at each
formation date inside the model.

Standard Compustat is updated historically and is not revision/as-reported
safe.  This script is therefore release-date-aware, not true vintage PIT.  It
flags RDQ fallbacks and supports an extra availability lag for sensitivity
tests.  A robust signal must survive reruns with 30- and 60-day extra lags.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd


EPS = 1e-12


@dataclass(frozen=True)
class BuildConfig:
    rdq_fallback_days: int = 90
    base_information_lag_days: int = 1
    extra_information_lag_days: int = 0
    maximum_staleness_days: int = 150
    minimum_ttm_quarters: int = 4
    sue_history_quarters: int = 8
    sue_minimum_quarters: int = 4
    output_buffer_assets: int = 100
    # Raw CRSP DlyCap-style market cap inputs are in thousands of dollars;
    # matches build_structural_alpha_features.py's --market-cap-scale
    # default so the two builders agree on what "market cap" means.
    market_cap_scale: float = 1000.0


FEATURE_COLUMNS = (
    "gross_profitability",
    "return_on_assets",
    "cash_return_on_assets",
    "asset_growth",
    "sales_growth",
    "fundamental_momentum",
    "standardized_unexpected_earnings",
    "accruals_to_assets",
    "book_to_assets",
    "leverage",
    "working_capital_to_assets",
    "rd_intensity",
)


REQUIRED_COLUMNS = {
    "gvkey",
    "permno",
    "datadate",
    "rdq",
    "fyearq",
    "fqtr",
    "atq",
    "ltq",
    "ceqq",
    "saleq",
    "cogsq",
    "ibq",
    "actq",
    "lctq",
    "dlcq",
    "dlttq",
    "oancfy",
    "xrdq",
}


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.where(denominator.abs() > EPS)
    result = numerator / denominator
    return result.replace([np.inf, -np.inf], np.nan)


def _load_wide(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    frame.index = pd.DatetimeIndex(frame.index)
    frame.columns = frame.columns.astype(str)
    if frame.index.has_duplicates:
        raise ValueError(f"{path} contains duplicate dates.")
    return frame.sort_index()


def _quarter_from_ytd(group: pd.DataFrame, column: str) -> pd.Series:
    """Convert a Compustat year-to-date flow to a fiscal-quarter flow."""
    values = pd.to_numeric(group[column], errors="coerce")
    previous = values.shift(1)
    same_year = group["fyearq"].eq(group["fyearq"].shift(1))
    sequential = group["fqtr"].eq(group["fqtr"].shift(1) + 1)
    quarter = values.copy()
    difference_mask = same_year & sequential & group["fqtr"].gt(1)
    quarter.loc[difference_mask] = values.loc[difference_mask] - previous.loc[difference_mask]
    return quarter


def _rolling_sum(
    series: pd.Series,
    quarter_id: pd.Series,
    quarters: int,
) -> pd.Series:
    result = series.rolling(quarters, min_periods=quarters).sum()
    consecutive = quarter_id - quarter_id.shift(quarters - 1) == quarters - 1
    return result.where(consecutive)


def _build_group_features(group: pd.DataFrame, config: BuildConfig) -> pd.DataFrame:
    group = group.sort_values(["datadate", "available_date"]).copy()
    numeric = [
        "atq", "ltq", "ceqq", "saleq", "cogsq", "ibq", "actq", "lctq",
        "dlcq", "dlttq", "oancfy", "xrdq", "fyearq", "fqtr",
    ]
    for column in numeric:
        group[column] = pd.to_numeric(group[column], errors="coerce")

    quarter_id = group["fyearq"] * 4.0 + group["fqtr"]

    group["operating_cash_flow_q"] = _quarter_from_ytd(group, "oancfy")
    group["gross_profit_q"] = group["saleq"] - group["cogsq"]
    group["gross_profit_ttm"] = _rolling_sum(
        group["gross_profit_q"], quarter_id, config.minimum_ttm_quarters
    )
    group["income_ttm"] = _rolling_sum(
        group["ibq"], quarter_id, config.minimum_ttm_quarters
    )
    group["sales_ttm"] = _rolling_sum(
        group["saleq"], quarter_id, config.minimum_ttm_quarters
    )
    group["cash_flow_ttm"] = _rolling_sum(
        group["operating_cash_flow_q"], quarter_id, config.minimum_ttm_quarters
    )

    valid_year_lag = quarter_id - quarter_id.shift(4) == 4
    lag_assets = group["atq"].shift(4).where(valid_year_lag)
    average_assets = 0.5 * (group["atq"] + lag_assets)
    lag_sales_ttm = group["sales_ttm"].shift(4).where(valid_year_lag)

    group["gross_profitability"] = _safe_divide(
        group["gross_profit_ttm"], average_assets
    )
    group["return_on_assets"] = _safe_divide(group["income_ttm"], average_assets)
    group["cash_return_on_assets"] = _safe_divide(
        group["cash_flow_ttm"], average_assets
    )
    group["asset_growth"] = _safe_divide(group["atq"], lag_assets) - 1.0
    group["sales_growth"] = _safe_divide(group["sales_ttm"], lag_sales_ttm) - 1.0
    group["fundamental_momentum"] = (
        group["return_on_assets"]
        - group["return_on_assets"].shift(4).where(valid_year_lag)
    )

    seasonal_change = group["ibq"] - group["ibq"].shift(4).where(valid_year_lag)
    prior_scale = seasonal_change.shift(1).rolling(
        config.sue_history_quarters,
        min_periods=config.sue_minimum_quarters,
    ).std(ddof=1)
    group["standardized_unexpected_earnings"] = _safe_divide(
        seasonal_change, prior_scale
    )
    group["accruals_to_assets"] = _safe_divide(
        group["income_ttm"] - group["cash_flow_ttm"], average_assets
    )
    group["book_to_assets"] = _safe_divide(group["ceqq"], group["atq"])
    group["leverage"] = _safe_divide(
        group["dlcq"].fillna(0.0) + group["dlttq"].fillna(0.0), group["atq"]
    )
    group["working_capital_to_assets"] = _safe_divide(
        group["actq"] - group["lctq"], group["atq"]
    )
    group["rd_intensity"] = _safe_divide(group["xrdq"], group["saleq"].abs())
    return group


def prepare_quarterly_features(
    fundamentals: pd.DataFrame,
    permitted_assets: set[str],
    config: BuildConfig,
) -> tuple[pd.DataFrame, dict[str, object]]:
    columns = {column.lower(): column for column in fundamentals.columns}
    missing = sorted(REQUIRED_COLUMNS.difference(columns))
    if missing:
        raise ValueError(f"Fundamentals file is missing required columns: {missing}")
    fundamentals = fundamentals.rename(columns={value: key for key, value in columns.items()})
    fundamentals["permno"] = (
        pd.to_numeric(fundamentals["permno"], errors="coerce").astype("Int64").astype(str)
    )
    fundamentals = fundamentals[fundamentals["permno"].isin(permitted_assets)].copy()
    fundamentals["datadate"] = pd.to_datetime(fundamentals["datadate"], errors="coerce")
    fundamentals["rdq"] = pd.to_datetime(fundamentals["rdq"], errors="coerce")
    rdq_fallback = fundamentals["rdq"].isna()
    fallback_date = fundamentals["datadate"] + pd.to_timedelta(
        config.rdq_fallback_days, unit="D"
    )
    if "available_date" in fundamentals:
        supplied = pd.to_datetime(fundamentals["available_date"], errors="coerce")
    else:
        supplied = pd.Series(pd.NaT, index=fundamentals.index)
    base_availability = fundamentals["rdq"].fillna(fallback_date)
    # Recompute from source fields rather than trusting an earlier availability
    # column; this makes sensitivity lags reproducible from one raw file.
    fundamentals["available_date"] = base_availability + pd.to_timedelta(
        config.base_information_lag_days + config.extra_information_lag_days,
        unit="D",
    )
    fundamentals["rdq_fallback"] = rdq_fallback.astype(int)
    fundamentals = fundamentals.dropna(subset=["gvkey", "permno", "datadate", "available_date"])
    fundamentals = fundamentals.sort_values(
        ["gvkey", "permno", "datadate", "available_date"]
    )

    pieces = []
    for _, group in fundamentals.groupby(["gvkey", "permno"], sort=False):
        pieces.append(_build_group_features(group, config))
    featured = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    if featured.empty:
        raise ValueError("No Compustat observations overlap the CRSP columns.")
    featured = featured.sort_values(["permno", "available_date", "datadate"])
    duplicate_availability = featured.duplicated(["permno", "available_date"], keep=False)
    featured = featured.drop_duplicates(["permno", "available_date"], keep="last")
    keep = [
        "permno", "gvkey", "datadate", "available_date", "rdq_fallback",
        # Raw common equity, kept alongside the derived FEATURE_COLUMNS ratios
        # so align_to_weekly_universe can compute book_to_market against a
        # lagged CRSP market cap (a cross-source ratio the Compustat-only
        # feature set can't express on its own).
        "ceqq",
        *FEATURE_COLUMNS,
    ]
    metadata = {
        "overlap_rows": int(len(featured)),
        "overlap_permnos": int(featured["permno"].nunique()),
        "rdq_fallback_fraction": float(featured["rdq_fallback"].mean()),
        "duplicate_availability_rows_removed": int(duplicate_availability.sum()),
        "supplied_available_date_disagreement_rows": int(
            (
                supplied.notna()
                & (supplied.dt.normalize() != fundamentals["available_date"].dt.normalize())
            ).sum()
        ) if len(supplied) == len(fundamentals) else None,
    }
    return featured[keep], metadata


def align_to_weekly_universe(
    quarterly: pd.DataFrame,
    weekly_index: pd.DatetimeIndex,
    pit: pd.DataFrame,
    destination: Path,
    config: BuildConfig,
    market_caps: pd.DataFrame | None = None,
) -> dict[str, object]:
    temporary = destination.with_name(destination.name + ".partial")
    temporary.unlink(missing_ok=True)
    quarterly_groups = {
        asset: group.sort_values("available_date").copy()
        for asset, group in quarterly.groupby("permno", sort=False)
    }
    assets = [asset for asset in pit.columns if asset in quarterly_groups]
    buffer: list[pd.DataFrame] = []
    total_rows = 0
    fallback_rows = 0
    earliest = None
    latest = None

    def flush() -> None:
        nonlocal buffer
        if not buffer:
            return
        block = pd.concat(buffer, ignore_index=True)
        block.to_csv(
            temporary,
            mode="a",
            header=not temporary.exists(),
            index=False,
            compression="gzip" if destination.suffix == ".gz" else None,
        )
        buffer = []

    for number, asset in enumerate(assets, start=1):
        membership = pit[asset].fillna(False).astype(bool)
        dates = weekly_index[membership.to_numpy()]
        if len(dates) == 0:
            continue
        left = pd.DataFrame({"formation_date": dates})
        source = quarterly_groups[asset]
        aligned = pd.merge_asof(
            left.sort_values("formation_date"),
            source.sort_values("available_date"),
            left_on="formation_date",
            right_on="available_date",
            direction="backward",
            allow_exact_matches=True,
        )
        aligned["permno"] = asset
        aligned["age_days"] = (
            aligned["formation_date"] - aligned["available_date"]
        ).dt.days
        aligned = aligned[
            aligned["available_date"].notna()
            & aligned["age_days"].between(0, config.maximum_staleness_days)
        ].copy()
        if aligned.empty:
            continue
        if (aligned["available_date"] > aligned["formation_date"]).any():
            raise AssertionError("Future Compustat information entered a formation row.")
        if market_caps is not None and asset in market_caps.columns:
            # book_to_market mixes Compustat book equity (ceqq, $ millions)
            # with a lagged CRSP market cap observed as of each formation
            # date -- i.e. the market cap known/PIT-eligible at the time,
            # not a same-quarter Compustat-only proxy. market_caps carries
            # raw CRSP-style units (thousands), same convention as
            # build_structural_alpha_features.py's market_cap_usd.
            raw_cap = aligned["formation_date"].map(market_caps[asset])
            market_cap_usd = raw_cap.astype(float) * config.market_cap_scale
            book_equity_usd = aligned["ceqq"].astype(float) * 1_000_000.0
            aligned["book_to_market"] = _safe_divide(book_equity_usd, market_cap_usd)
        else:
            aligned["book_to_market"] = np.nan

        selected = [
            "formation_date", "permno", "gvkey", "datadate", "available_date",
            "age_days", "rdq_fallback", "book_to_market", *FEATURE_COLUMNS,
        ]
        aligned = aligned[selected]
        buffer.append(aligned)
        total_rows += len(aligned)
        fallback_rows += int(aligned["rdq_fallback"].sum())
        earliest = aligned["formation_date"].min() if earliest is None else min(
            earliest, aligned["formation_date"].min()
        )
        latest = aligned["formation_date"].max() if latest is None else max(
            latest, aligned["formation_date"].max()
        )
        if len(buffer) >= config.output_buffer_assets:
            flush()
        if number % 250 == 0 or number == len(assets):
            print(f"aligned {number:,}/{len(assets):,} assets; rows={total_rows:,}")
    flush()
    if not temporary.exists():
        raise ValueError("No weekly feature rows were produced.")
    temporary.replace(destination)
    return {
        "weekly_feature_rows": int(total_rows),
        "weekly_feature_permnos": int(len(assets)),
        "weekly_fallback_fraction": float(fallback_rows / max(total_rows, 1)),
        "first_formation_date": str(earliest.date()) if earliest is not None else None,
        "last_formation_date": str(latest.date()) if latest is not None else None,
    }


def _self_test() -> None:
    dates = pd.date_range("2018-03-31", periods=16, freq="QE")
    fundamentals = pd.DataFrame(
        {
            "gvkey": "001000",
            "permno": 10001,
            "datadate": dates,
            "rdq": dates + pd.Timedelta(days=40),
            "fyearq": dates.year,
            "fqtr": np.tile([1, 2, 3, 4], 4),
            "atq": np.linspace(100, 130, 16),
            "ltq": 40.0,
            "ceqq": 60.0,
            "saleq": np.linspace(25, 35, 16),
            "cogsq": 15.0,
            "ibq": np.array(
                [2.0, 2.4, 2.2, 2.8, 2.3, 2.9, 2.5, 3.4,
                 2.8, 3.1, 3.2, 3.7, 3.0, 3.8, 3.4, 4.4]
            ),
            "actq": 50.0,
            "lctq": 20.0,
            "dlcq": 5.0,
            "dlttq": 20.0,
            "oancfy": np.tile([3.0, 7.0, 12.0, 18.0], 4),
            "xrdq": 1.0,
        }
    )
    config = BuildConfig()
    quarterly, metadata = prepare_quarterly_features(fundamentals, {"10001"}, config)
    assert metadata["overlap_rows"] == 16
    assert quarterly["return_on_assets"].notna().sum() > 0
    assert quarterly["standardized_unexpected_earnings"].notna().sum() > 0
    assert (quarterly["available_date"] > quarterly["datadate"]).all()
    weekly_index = pd.date_range("2019-01-04", "2022-12-30", freq="W-FRI")
    pit = pd.DataFrame(True, index=weekly_index, columns=["10001"])
    with TemporaryDirectory() as directory:
        destination = Path(directory) / "features.csv.gz"
        weekly_metadata = align_to_weekly_universe(
            quarterly, weekly_index, pit, destination, config
        )
        aligned = pd.read_csv(
            destination,
            parse_dates=["formation_date", "datadate", "available_date"],
        )
        assert weekly_metadata["weekly_feature_rows"] == len(aligned)
        assert (aligned["available_date"] <= aligned["formation_date"]).all()
        assert aligned.duplicated(["formation_date", "permno"]).sum() == 0
    print("self-test passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build weekly release-aware Compustat features for CRSP PERMNOs."
    )
    parser.add_argument("--fundamentals", help="Long Compustat quarterly CSV.")
    parser.add_argument("--returns", help="Wide weekly CRSP returns CSV.")
    parser.add_argument("--pit", help="Matching PIT universe CSV.")
    parser.add_argument(
        "--output",
        default="wrds_full/compustat_pit_features_long.csv.gz",
    )
    parser.add_argument("--rdq-fallback-days", type=int, default=90)
    parser.add_argument("--base-information-lag-days", type=int, default=1)
    parser.add_argument("--extra-information-lag-days", type=int, default=0)
    parser.add_argument("--maximum-staleness-days", type=int, default=150)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        _self_test()
        return
    if not args.fundamentals or not args.returns or not args.pit:
        raise ValueError("--fundamentals, --returns, and --pit are required.")
    config = BuildConfig(
        rdq_fallback_days=args.rdq_fallback_days,
        base_information_lag_days=args.base_information_lag_days,
        extra_information_lag_days=args.extra_information_lag_days,
        maximum_staleness_days=args.maximum_staleness_days,
    )
    returns = _load_wide(args.returns)
    pit = _load_wide(args.pit).reindex(
        index=returns.index, columns=returns.columns
    ).fillna(0.0).astype(bool)
    fundamentals = pd.read_csv(
        args.fundamentals,
        low_memory=False,
        parse_dates=[column for column in ("datadate", "rdq", "available_date")
                     if column in pd.read_csv(args.fundamentals, nrows=0).columns],
    )
    quarterly, quarterly_metadata = prepare_quarterly_features(
        fundamentals, set(returns.columns), config
    )
    destination = Path(args.output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    weekly_metadata = align_to_weekly_universe(
        quarterly, returns.index, pit, destination, config
    )
    metadata = {
        "methodology": "release_date_aware_not_revision_safe",
        "features": list(FEATURE_COLUMNS),
        "config": asdict(config),
        "input_fundamental_rows": int(len(fundamentals)),
        "crsp_weeks": int(len(returns)),
        "crsp_assets": int(returns.shape[1]),
        **quarterly_metadata,
        **weekly_metadata,
    }
    metadata_path = destination.with_name(destination.name + ".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    print(f"saved: {destination}")


if __name__ == "__main__":
    main()
