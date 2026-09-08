from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CRSPQueryConfig:
    # None means auto-discover. These match the schemas exposed by the
    # project's WRDS account, in preference order.
    library: str | None = None
    preferred_libraries: tuple[str, ...] = ("crsp", "crsp_a_stock")
    preferred_tables: tuple[str, ...] = ("dsf_v2", "stkdlysecuritydata")
    security_info_tables: tuple[str, ...] = ("stksecurityinfohist",)
    common_stocks_only: bool = True
    # Optional point-in-time filters. Explicit security-class filters are
    # mutually exclusive with common_stocks_only; ticker filtering can be
    # combined with either universe definition.
    tickers: tuple[str, ...] = ()
    share_types: tuple[str, ...] = ()
    security_types: tuple[str, ...] = ()
    security_subtypes: tuple[str, ...] = ()
    us_incorporated_values: tuple[str, ...] = ()
    issuer_types: tuple[str, ...] = ()


@dataclass(frozen=True)
class CRSPWeeklyConfig:
    week_frequency: str = "W-FRI"
    minimum_history_days: int = 252
    liquidity_window_days: int = 63
    minimum_price: float | None = 5.0
    minimum_market_cap: float | None = None
    maximum_market_cap: float | None = None
    minimum_median_dollar_volume: float | None = None
    maximum_beta: float | None = None
    beta_lookback_weeks: int = 52
    beta_minimum_observations: int = 26
    top_n_by_market_cap: int | None = None
    ranking_lag_months: int = 1
    drop_incomplete_final_week: bool = True


@dataclass(frozen=True)
class CRSPChunkConfig:
    """Controls resumable WRDS downloads.

    Intermediate chunk boundaries are moved to Fridays so a weekly return is
    never split across two downloaded chunks.
    """

    months_per_chunk: int = 12
    cache_dir: str | Path = "wrds_cache"
    resume: bool = True


@dataclass(frozen=True)
class CRSPWeeklyBundle:
    returns: pd.DataFrame
    pit_universe: pd.DataFrame
    metadata: pd.DataFrame
    diagnostics: dict[str, Any]


_COLUMN_CANDIDATES = {
    "permno": ("permno",),
    "date": ("dlycaldt", "date"),
    "ret": ("dlyret", "ret"),
    "price": ("dlyprc", "prc"),
    "market_cap": ("dlycap", "marketcap", "mktcap"),
    "volume": ("dlyvol", "vol"),
    "ticker": ("ticker", "tradingticker"),
    "company_name": ("issuername", "comnam"),
    "share_type": ("sharetype",),
    "security_type": ("securitytype",),
    "security_subtype": ("securitysubtype",),
    "us_incorporated": ("usincflg",),
    "issuer_type": ("issuertype",),
    "sec_info_start": ("secinfostartdt",),
    "sec_info_end": ("secinfoenddt",),
    "sic_code": ("siccd",),
    "naics_code": ("naics",),
    "icb_industry": ("icbindustry",),
    "ues_industry": ("uesindustry",),
    "primary_exchange": ("primaryexch",),
    "trading_status": ("tradingstatusflg",),
}


def _safe_identifier(value: str) -> str:
    text = str(value)
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", text):
        raise ValueError(f"Unsafe SQL identifier: {value!r}")
    return text


def _description_columns(description: Any) -> list[str]:
    if isinstance(description, pd.DataFrame):
        for candidate in ("name", "column_name", "variable"):
            if candidate in description.columns:
                return description[candidate].astype(str).tolist()
        if description.index.dtype == object:
            return description.index.astype(str).tolist()
    if isinstance(description, (list, tuple)):
        columns = []
        for item in description:
            if isinstance(item, dict):
                columns.append(str(item.get("name") or item.get("column_name")))
            else:
                columns.append(str(item))
        return [column for column in columns if column and column != "None"]
    raise TypeError("Unsupported WRDS table description format.")


def _resolve_columns(
    columns: list[str],
    required: tuple[str, ...] = ("permno", "date", "ret"),
) -> dict[str, str]:
    lookup = {column.lower(): column for column in columns}
    resolved: dict[str, str] = {}
    for canonical, candidates in _COLUMN_CANDIDATES.items():
        for candidate in candidates:
            if candidate.lower() in lookup:
                resolved[canonical] = lookup[candidate.lower()]
                break
    missing = [name for name in required if name not in resolved]
    if missing:
        raise ValueError(f"CRSP table is missing required fields: {missing}")
    return resolved


def resolve_crsp_table(
    connection: Any,
    config: CRSPQueryConfig | None = None,
) -> tuple[str, str, dict[str, str]]:
    """Select the first accessible CIZ daily table and map its columns."""
    cfg = config or CRSPQueryConfig()
    if cfg.library:
        libraries = [str(cfg.library)]
    else:
        available_libraries = {
            str(library).lower(): str(library)
            for library in connection.list_libraries()
        }
        libraries = [
            available_libraries[library.lower()]
            for library in cfg.preferred_libraries
            if library.lower() in available_libraries
        ]
    if not libraries:
        raise RuntimeError(
            "No supported CRSP stock schema is accessible. Expected one of: "
            + ", ".join(cfg.preferred_libraries)
        )

    errors = []
    for library in libraries:
        try:
            available = {
                str(table).lower(): str(table)
                for table in connection.list_tables(library=library)
            }
        # WRDS adapters surface account, network, and database errors through
        # different exception classes; collect them so discovery can continue.
        except Exception as exc:  # noqa: BLE001  # pragma: no cover
            errors.append(f"{library}: {exc}")
            continue

        for preferred in cfg.preferred_tables:
            table = available.get(preferred.lower())
            if table is None:
                continue
            try:
                description = connection.describe_table(
                    library=library,
                    table=table,
                )
                return (
                    library,
                    table,
                    _resolve_columns(_description_columns(description)),
                )
            except Exception as exc:  # noqa: BLE001  # pragma: no cover
                errors.append(f"{library}.{table}: {exc}")
    detail = "; ".join(errors) if errors else "no preferred table was listed"
    raise RuntimeError(
        f"Could not resolve a CRSP CIZ daily table in {libraries!r}: {detail}. "
        "Inspect connection.list_libraries() and list_tables()."
    )


def fetch_crsp_ciz_daily(
    connection: Any,
    start: str | date,
    end: str | date,
    config: CRSPQueryConfig | None = None,
    _resolved: tuple[str, str, dict[str, str]] | None = None,
) -> pd.DataFrame:
    """Download research fields from CRSP CIZ without duplicating delisting returns.

    CIZ ``DlyRet`` is treated as the total return supplied by CRSP.  No legacy
    ``DLRET`` multiplication is performed.
    """
    cfg = config or CRSPQueryConfig()
    explicit_classification = {
        "share_type": tuple(cfg.share_types),
        "security_type": tuple(cfg.security_types),
        "security_subtype": tuple(cfg.security_subtypes),
        "us_incorporated": tuple(cfg.us_incorporated_values),
        "issuer_type": tuple(cfg.issuer_types),
    }
    if cfg.common_stocks_only and any(explicit_classification.values()):
        raise ValueError(
            "Explicit CRSP security-class filters require common_stocks_only=False."
        )

    library, table, columns = _resolved or resolve_crsp_table(connection, cfg)
    library = _safe_identifier(library)
    table = _safe_identifier(table)

    source_expressions = {
        canonical: f"d.{_safe_identifier(source)}"
        for canonical, source in columns.items()
        if canonical not in {"sec_info_start", "sec_info_end"}
    }
    join_clause = ""
    info_table: str | None = None

    classification_fields = (
        "share_type",
        "security_type",
        "security_subtype",
        "us_incorporated",
        "issuer_type",
    )
    required_source_fields = set()
    if cfg.common_stocks_only:
        required_source_fields.update(classification_fields)
    else:
        required_source_fields.update(
            field for field, values in explicit_classification.items() if values
        )
    if cfg.tickers:
        required_source_fields.add("ticker")

    missing_required = [
        field for field in required_source_fields if field not in source_expressions
    ]
    if missing_required:
        available = {
            str(candidate).lower(): str(candidate)
            for candidate in connection.list_tables(library=library)
        }
        info_table = next(
            (
                available[candidate.lower()]
                for candidate in cfg.security_info_tables
                if candidate.lower() in available
            ),
            None,
        )
        if info_table is not None:
            info_description = connection.describe_table(
                library=library,
                table=info_table,
            )
            info_columns = _resolve_columns(
                _description_columns(info_description),
                required=("permno", "sec_info_start", "sec_info_end"),
            )
            for canonical in (
                "ticker",
                "company_name",
                *classification_fields,
            ):
                if canonical not in source_expressions and canonical in info_columns:
                    source_expressions[canonical] = (
                        f"i.{_safe_identifier(info_columns[canonical])}"
                    )
            safe_info_table = _safe_identifier(info_table)
            join_clause = (
                f" JOIN {library}.{safe_info_table} i"
                f" ON d.{_safe_identifier(columns['permno'])} = "
                f"i.{_safe_identifier(info_columns['permno'])}"
                f" AND d.{_safe_identifier(columns['date'])} BETWEEN "
                f"i.{_safe_identifier(info_columns['sec_info_start'])} AND "
                f"i.{_safe_identifier(info_columns['sec_info_end'])}"
            )

        still_missing = [
            field for field in required_source_fields if field not in source_expressions
        ]
        if still_missing:
            raise RuntimeError(
                "Cannot enforce the requested point-in-time CRSP universe; "
                f"missing classification fields {still_missing} in {library}.{table} "
                "and no complete date-effective stkSecurityInfoHist mapping was found. "
                "Do not weaken the filter unless an unfiltered security universe is "
                "intentional."
            )

    selected = [
        f"{expression} AS {_safe_identifier(canonical)}"
        for canonical, expression in source_expressions.items()
    ]

    query_parameters: dict[str, Any] = {
        "start": str(start),
        "end": str(end),
    }
    filters = [f"d.{_safe_identifier(columns['date'])} BETWEEN %(start)s AND %(end)s"]
    if cfg.common_stocks_only:
        equality_filters = {
            "share_type": "NS",
            "security_type": "EQTY",
            "security_subtype": "COM",
            "us_incorporated": "Y",
        }
        for canonical, value in equality_filters.items():
            filters.append(f"{source_expressions[canonical]} = '{value}'")
        if "issuer_type" in source_expressions:
            filters.append(f"{source_expressions['issuer_type']} IN ('ACOR', 'CORP')")
    else:
        for canonical, values in explicit_classification.items():
            if not values:
                continue
            placeholders = []
            for number, value in enumerate(values):
                parameter = f"filter_{canonical}_{number}"
                query_parameters[parameter] = str(value).upper()
                placeholders.append(f"%({parameter})s")
            filters.append(
                f"{source_expressions[canonical]} IN ({', '.join(placeholders)})"
            )

    if cfg.tickers:
        ticker_placeholders = []
        for number, ticker in enumerate(cfg.tickers):
            parameter = f"filter_ticker_{number}"
            query_parameters[parameter] = str(ticker).upper()
            ticker_placeholders.append(f"%({parameter})s")
        filters.append(
            f"UPPER({source_expressions['ticker']}) IN "
            f"({', '.join(ticker_placeholders)})"
        )

    query = (
        "SELECT "
        + ", ".join(selected)
        + f" FROM {library}.{table} d"
        + join_clause
        + " WHERE "
        + " AND ".join(filters)
        + f" ORDER BY d.{_safe_identifier(columns['permno'])}, "
        + f"d.{_safe_identifier(columns['date'])}"
    )
    daily = connection.raw_sql(
        query,
        params=query_parameters,
        date_cols=["date"],
    )
    out = standardize_crsp_daily(daily)
    out.attrs["wrds_library"] = library
    out.attrs["wrds_table"] = table
    if info_table is not None:
        out.attrs["wrds_security_info_table"] = info_table
    return out


def fetch_crsp_sector_history(
    connection: Any,
    start: str | date,
    end: str | date,
    config: CRSPQueryConfig | None = None,
) -> pd.DataFrame:
    """Download date-effective CRSP sector and security classifications.

    UES is preferred because CRSP defines it at a broad sector level. ICB is
    the second choice and a two-digit SIC group is the final fallback. The
    function fails closed when the account's security-history table exposes no
    date-effective industry field.
    """

    cfg = config or CRSPQueryConfig()
    library, _, _ = resolve_crsp_table(connection, cfg)
    safe_library = _safe_identifier(library)
    available = {
        str(candidate).lower(): str(candidate)
        for candidate in connection.list_tables(library=library)
    }
    info_table = next(
        (
            available[candidate.lower()]
            for candidate in cfg.security_info_tables
            if candidate.lower() in available
        ),
        None,
    )
    if info_table is None:
        raise RuntimeError(
            f"No date-effective CRSP security-history table found in {library}."
        )

    description = connection.describe_table(library=library, table=info_table)
    columns = _resolve_columns(
        _description_columns(description),
        required=("permno", "sec_info_start", "sec_info_end"),
    )
    sector_fields = [
        field
        for field in ("ues_industry", "icb_industry", "sic_code")
        if field in columns
    ]
    if not sector_fields:
        raise RuntimeError(
            f"{library}.{info_table} has no UESIndustry, ICBIndustry, or SICCD field."
        )

    optional = [
        field
        for field in (
            "ticker",
            "company_name",
            "share_type",
            "security_type",
            "security_subtype",
            "us_incorporated",
            "issuer_type",
            "primary_exchange",
            "trading_status",
            *sector_fields,
        )
        if field in columns
    ]
    selected_fields = ["permno", "sec_info_start", "sec_info_end", *optional]
    select_clause = ", ".join(
        f"i.{_safe_identifier(columns[field])} AS {_safe_identifier(field)}"
        for field in selected_fields
    )
    safe_info_table = _safe_identifier(info_table)
    query = (
        f"SELECT {select_clause} FROM {safe_library}.{safe_info_table} i "
        f"WHERE i.{_safe_identifier(columns['sec_info_end'])} >= %(start)s "
        f"AND i.{_safe_identifier(columns['sec_info_start'])} <= %(end)s "
        f"ORDER BY i.{_safe_identifier(columns['permno'])}, "
        f"i.{_safe_identifier(columns['sec_info_start'])}"
    )
    history = connection.raw_sql(
        query,
        params={"start": str(start), "end": str(end)},
        date_cols=["sec_info_start", "sec_info_end"],
    )
    if history.empty:
        raise ValueError("CRSP sector-history query returned no rows.")

    history = history.copy()
    history["permno"] = pd.to_numeric(history["permno"], errors="raise").astype("int64")
    history["sec_info_start"] = pd.to_datetime(
        history["sec_info_start"], errors="raise"
    )
    # Current securities can carry a database end date beyond pandas' 2262
    # timestamp limit. Storing that boundary as pandas' maximum date preserves
    # the intended open-ended interval for downstream as-of joins.
    history["sec_info_end"] = pd.to_datetime(
        history["sec_info_end"], errors="coerce"
    ).fillna(pd.Timestamp.max.normalize())
    for column in optional:
        history[column] = history[column].astype("string").str.strip()

    sector = pd.Series(pd.NA, index=history.index, dtype="string")
    if "ues_industry" in history:
        sector = history["ues_industry"].astype("string").str.strip()
        sector = sector.mask(sector.str.upper().isin({"", "NOAVAIL", "N/A", "NA"}))
    if "icb_industry" in history:
        icb = history["icb_industry"].astype("string").str.strip()
        icb = icb.mask(icb.str.upper().isin({"", "NOAVAIL", "N/A", "NA"}))
        sector = sector.fillna(icb)
    if "sic_code" in history:
        sic = history["sic_code"].str.extract(r"(\d{2})", expand=False)
        sector = sector.fillna("SIC_" + sic)
    history["sector"] = sector
    history = history.loc[history["sector"].notna()].copy()
    if history.empty:
        raise ValueError("CRSP sector history contains no usable sector labels.")
    return history.sort_values(["permno", "sec_info_start"]).reset_index(drop=True)


def _date_chunks(
    start: str | date,
    end: str | date,
    months_per_chunk: int,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    first = pd.Timestamp(start).normalize()
    last = pd.Timestamp(end).normalize()
    if first > last:
        raise ValueError("start must be on or before end.")
    if int(months_per_chunk) < 1:
        raise ValueError("months_per_chunk must be at least 1.")

    chunks: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cursor = first
    while cursor <= last:
        proposed = min(
            cursor + pd.DateOffset(months=int(months_per_chunk)) - pd.Timedelta(days=1),
            last,
        )
        if proposed < last:
            # Close intermediate chunks on Friday. The next chunk begins on
            # Saturday, preventing a W-FRI return from being split.
            proposed -= pd.Timedelta(days=(proposed.weekday() - 4) % 7)
            if proposed < cursor:
                proposed = min(cursor + pd.Timedelta(days=6), last)
        chunks.append((cursor, proposed))
        cursor = proposed + pd.Timedelta(days=1)
    return chunks


def iter_crsp_ciz_daily_chunks(
    connection: Any,
    start: str | date,
    end: str | date,
    query_config: CRSPQueryConfig | None = None,
    chunk_config: CRSPChunkConfig | None = None,
) -> Iterator[pd.DataFrame]:
    """Yield validated CRSP daily chunks with optional restartable caching."""
    query_cfg = query_config or CRSPQueryConfig()
    chunk_cfg = chunk_config or CRSPChunkConfig()
    resolved = resolve_crsp_table(connection, query_cfg)
    library, table, _ = resolved

    cache_dir = Path(chunk_cfg.cache_dir).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    chunks = _date_chunks(start, end, chunk_cfg.months_per_chunk)

    for chunk_number, (chunk_start, chunk_end) in enumerate(chunks, start=1):
        custom_filters = {
            "tickers": query_cfg.tickers,
            "share_types": query_cfg.share_types,
            "security_types": query_cfg.security_types,
            "security_subtypes": query_cfg.security_subtypes,
            "us_incorporated_values": query_cfg.us_incorporated_values,
            "issuer_types": query_cfg.issuer_types,
        }
        if any(custom_filters.values()):
            serialized = json.dumps(custom_filters, sort_keys=True).encode("utf-8")
            filter_tag = "_filter" + hashlib.sha256(serialized).hexdigest()[:12]
        else:
            filter_tag = ""
        cache_name = (
            f"{library}_{table}_{chunk_start:%Y%m%d}_{chunk_end:%Y%m%d}_"
            f"common{int(query_cfg.common_stocks_only)}{filter_tag}.csv.gz"
        )
        cache_path = cache_dir / cache_name
        cache_hit = bool(chunk_cfg.resume and cache_path.exists())

        if cache_hit:
            daily = pd.read_csv(cache_path, parse_dates=["date"])
            daily = standardize_crsp_daily(daily)
        else:
            daily = fetch_crsp_ciz_daily(
                connection,
                chunk_start.date(),
                chunk_end.date(),
                query_cfg,
                _resolved=resolved,
            )
            temporary = cache_path.with_name(cache_path.name + ".tmp")
            daily.to_csv(temporary, index=False, compression="gzip")
            temporary.replace(cache_path)

        daily.attrs["wrds_library"] = library
        daily.attrs["wrds_table"] = table
        daily.attrs["chunk_number"] = chunk_number
        daily.attrs["chunk_count"] = len(chunks)
        daily.attrs["chunk_start"] = str(chunk_start.date())
        daily.attrs["chunk_end"] = str(chunk_end.date())
        daily.attrs["chunk_cache_hit"] = cache_hit
        yield daily


def standardize_crsp_daily(daily: pd.DataFrame) -> pd.DataFrame:
    required = {"permno", "date", "ret"}
    missing = required.difference(daily.columns)
    if missing:
        raise ValueError(f"CRSP daily data is missing columns: {sorted(missing)}")

    out = daily.copy()
    out["date"] = pd.to_datetime(out["date"], errors="raise")
    out["permno"] = pd.to_numeric(out["permno"], errors="raise").astype("int64")
    for column in ("ret", "price", "market_cap", "volume"):
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    if "price" in out.columns:
        out["price"] = out["price"].abs()
    out = out.sort_values(["permno", "date"]).reset_index(drop=True)
    if out.duplicated(["permno", "date"]).any():
        examples = (
            out.loc[
                out.duplicated(["permno", "date"], keep=False),
                ["permno", "date"],
            ]
            .head()
            .to_dict("records")
        )
        raise ValueError(f"Duplicate CRSP security-date rows: {examples}")
    if (out["ret"].dropna() < -1.0 - 1e-12).any():
        raise ValueError("CRSP returns below -100% were found.")
    return out


def _compound_returns(values: pd.Series) -> float:
    clean = values.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if clean.empty:
        return np.nan
    if (clean < -1.0 - 1e-12).any():
        raise ValueError("Cannot compound a return below -100%.")
    return float(np.prod(1.0 + clean.to_numpy()) - 1.0)


def build_crsp_weekly_bundle(
    daily: pd.DataFrame,
    config: CRSPWeeklyConfig | None = None,
) -> CRSPWeeklyBundle:
    """Create weekly research inputs from one in-memory daily frame."""
    return build_crsp_weekly_bundle_from_chunks([daily], config)


def _rank_lagged_monthly_top_n(
    weekly: pd.DataFrame,
    top_n: int,
    lag_months: int,
) -> tuple[pd.DataFrame, set[str]]:
    """Apply a deterministic, date-effective market-cap membership rule."""
    if top_n < 1:
        raise ValueError("top_n_by_market_cap must be at least 1.")
    if lag_months < 1:
        raise ValueError(
            "ranking_lag_months must be at least 1 to prevent look-ahead bias."
        )
    if "market_cap" not in weekly.columns:
        raise ValueError("top_n_by_market_cap requires a CRSP market-cap column.")

    ranked = weekly.copy()
    ranked["month"] = ranked["week"].dt.to_period("M")
    ranked["market_cap"] = pd.to_numeric(ranked["market_cap"], errors="coerce")

    # One observation per asset and formation month, using the final weekly
    # record available in that month. Only already-eligible securities rank.
    snapshots = (
        ranked.sort_values(["week", "asset"])
        .groupby(["month", "asset"], sort=True, observed=True)
        .tail(1)
    )
    snapshots = snapshots.loc[
        snapshots["eligible"]
        & snapshots["market_cap"].notna()
        & (snapshots["market_cap"] > 0)
    ].copy()
    snapshots = snapshots.sort_values(
        ["month", "market_cap", "asset"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    snapshots["rank"] = snapshots.groupby("month", observed=True).cumcount() + 1
    selected = snapshots.loc[snapshots["rank"] <= int(top_n), ["month", "asset"]]
    selected["effective_month"] = selected["month"] + int(lag_months)
    selected = selected[["effective_month", "asset"]].drop_duplicates()
    selected["selected"] = True

    ranked = ranked.merge(
        selected,
        left_on=["month", "asset"],
        right_on=["effective_month", "asset"],
        how="left",
        validate="many_to_one",
    )
    ranked["pit_eligible"] = ranked["eligible"] & ranked["selected"].eq(True)
    selected_assets = set(selected["asset"].astype(str))
    return ranked.drop(columns=["effective_month", "selected"]), selected_assets


def _apply_lagged_beta_filter(
    weekly: pd.DataFrame,
    maximum_beta: float,
    lookback_weeks: int,
    minimum_observations: int,
) -> pd.DataFrame:
    """Require a lagged rolling beta below a fixed PIT threshold.

    The market proxy is the cross-sectional return weighted by each security's
    previous-week market cap. Asset and market returns are shifted one week
    before beta estimation, so the formation week's return never determines
    its own eligibility.
    """

    if "market_cap" not in weekly:
        raise ValueError("The low-beta filter requires CRSP market capitalization.")
    if lookback_weeks < 4:
        raise ValueError("beta_lookback_weeks must be at least 4.")
    if not 3 <= minimum_observations <= lookback_weeks:
        raise ValueError(
            "beta_minimum_observations must be between 3 and beta_lookback_weeks."
        )

    ranked = weekly.sort_values(["asset", "week"]).copy()
    ranked["_lagged_market_cap"] = ranked.groupby(
        "asset", observed=True
    )["market_cap"].shift(1)
    # Build the benchmark from the full downloaded common-stock cross-section,
    # not merely the micro-cap candidates. Otherwise "low beta" would mean
    # low beta relative to micro-caps rather than to the broad equity market.
    usable = ranked["ret"].notna() & ranked["_lagged_market_cap"].gt(0.0)
    weighted_return = (
        ranked["ret"] * ranked["_lagged_market_cap"]
    ).where(usable)
    numerator = weighted_return.groupby(ranked["week"]).sum(min_count=1)
    denominator = (
        ranked["_lagged_market_cap"]
        .where(usable)
        .groupby(ranked["week"])
        .sum(min_count=1)
    )
    market_return = numerator / denominator.replace(0.0, np.nan)
    ranked["_market_return"] = ranked["week"].map(market_return)

    beta = pd.Series(np.nan, index=ranked.index, dtype=float)
    for positions in ranked.groupby("asset", observed=True).groups.values():
        asset_return = ranked.loc[positions, "ret"].shift(1)
        lagged_market = ranked.loc[positions, "_market_return"].shift(1)
        covariance = asset_return.rolling(
            lookback_weeks, min_periods=minimum_observations
        ).cov(lagged_market)
        market_variance = lagged_market.rolling(
            lookback_weeks, min_periods=minimum_observations
        ).var(ddof=1)
        beta.loc[positions] = covariance / market_variance.replace(0.0, np.nan)
    ranked["lagged_beta"] = beta
    ranked["eligible"] = (
        ranked["eligible"]
        & ranked["lagged_beta"].notna()
        & ranked["lagged_beta"].le(float(maximum_beta))
    )
    return ranked.drop(columns=["_lagged_market_cap", "_market_return"])


def build_crsp_weekly_bundle_from_chunks(
    daily_chunks: Iterable[pd.DataFrame],
    config: CRSPWeeklyConfig | None = None,
) -> CRSPWeeklyBundle:
    """Incrementally aggregate daily chunks and build a lagged PIT universe.

    Daily chunks are discarded after weekly aggregation. Only a short rolling
    liquidity tail and cumulative observation counts are retained, avoiding a
    full-history daily concatenation in memory.
    """
    cfg = config or CRSPWeeklyConfig()
    if int(cfg.minimum_history_days) < 1:
        raise ValueError("minimum_history_days must be at least 1.")
    if int(cfg.liquidity_window_days) < 1:
        raise ValueError("liquidity_window_days must be at least 1.")
    if (
        cfg.minimum_market_cap is not None
        and cfg.maximum_market_cap is not None
        and cfg.minimum_market_cap > cfg.maximum_market_cap
    ):
        raise ValueError("minimum_market_cap cannot exceed maximum_market_cap.")

    weekly_parts: list[pd.DataFrame] = []
    metadata_parts: list[pd.DataFrame] = []
    history_counts: dict[int, int] = {}
    liquidity_tail = pd.DataFrame()
    total_daily_rows = 0
    chunk_count = 0
    cache_hits = 0
    last_observation: pd.Timestamp | None = None
    source_attrs: dict[str, Any] = {}

    for daily in daily_chunks:
        chunk_count += 1
        raw_attrs = dict(getattr(daily, "attrs", {}))
        if not source_attrs:
            source_attrs = raw_attrs
        cache_hits += int(bool(raw_attrs.get("chunk_cache_hit", False)))

        x = standardize_crsp_daily(daily)
        if x.empty:
            continue
        total_daily_rows += len(x)
        observed_max = x["date"].max().normalize()
        last_observation = (
            observed_max
            if last_observation is None
            else max(last_observation, observed_max)
        )
        x["asset"] = x["permno"].astype(str)

        prior_counts = x["permno"].map(history_counts).fillna(0).astype("int64")
        x["history_days"] = (
            x.groupby("permno", sort=False).cumcount() + 1 + prior_counts
        )
        new_counts = x.groupby("permno", sort=False).size()
        for permno, count in new_counts.items():
            key = int(permno)
            history_counts[key] = history_counts.get(key, 0) + int(count)

        if cfg.minimum_median_dollar_volume is not None:
            if not {"price", "volume"}.issubset(x.columns):
                raise ValueError(
                    "minimum_median_dollar_volume requires price and volume columns."
                )
            current = x.copy()
            current["_current_chunk"] = True
            if liquidity_tail.empty:
                combined = current
            else:
                tail = liquidity_tail.copy()
                tail["_current_chunk"] = False
                combined = pd.concat([tail, current], ignore_index=True, sort=False)
            combined = combined.sort_values(["permno", "date"]).reset_index(drop=True)
            combined["dollar_volume"] = combined["price"] * combined["volume"]
            minimum_periods = max(5, int(cfg.liquidity_window_days) // 2)
            combined["median_dollar_volume"] = combined.groupby("permno", sort=False)[
                "dollar_volume"
            ].transform(
                lambda values, min_obs=minimum_periods: values.rolling(
                    int(cfg.liquidity_window_days),
                    min_periods=min_obs,
                ).median()
            )
            x = combined.loc[combined["_current_chunk"]].drop(
                columns=["_current_chunk"]
            )
            tail_length = max(0, int(cfg.liquidity_window_days) - 1)
            liquidity_tail = (
                combined.groupby("permno", sort=False, group_keys=False)
                .tail(tail_length)
                .drop(columns=["_current_chunk"], errors="ignore")
            )

        x["week"] = (
            x["date"].dt.to_period(cfg.week_frequency).dt.end_time.dt.normalize()
        )
        eligible = x["history_days"] >= int(cfg.minimum_history_days)

        if cfg.minimum_price is not None:
            if "price" not in x.columns:
                raise ValueError("minimum_price requires a CRSP price column.")
            eligible &= x["price"] >= float(cfg.minimum_price)
        if cfg.minimum_market_cap is not None:
            if "market_cap" not in x.columns:
                raise ValueError("minimum_market_cap requires a market-cap column.")
            eligible &= x["market_cap"] >= float(cfg.minimum_market_cap)
        if cfg.maximum_market_cap is not None:
            if "market_cap" not in x.columns:
                raise ValueError("maximum_market_cap requires a market-cap column.")
            eligible &= x["market_cap"] <= float(cfg.maximum_market_cap)
        if cfg.minimum_median_dollar_volume is not None:
            eligible &= x["median_dollar_volume"] >= float(
                cfg.minimum_median_dollar_volume
            )

        x["eligible"] = eligible.fillna(False).astype(bool)
        grouped = x.groupby(["week", "asset"], sort=True, observed=True)
        part = pd.concat(
            [
                grouped["ret"].apply(_compound_returns).rename("ret"),
                grouped["eligible"].last().rename("eligible"),
            ],
            axis=1,
        ).reset_index()
        if "market_cap" in x.columns:
            cap = grouped["market_cap"].last().rename("market_cap").reset_index()
            part = part.merge(
                cap, on=["week", "asset"], how="left", validate="one_to_one"
            )
        weekly_parts.append(part)

        metadata_columns = [
            column
            for column in (
                "permno",
                "date",
                "ticker",
                "company_name",
                "ues_industry",
                "icb_industry",
                "sic_code",
                "naics_code",
                "primary_exchange",
                "trading_status",
            )
            if column in x.columns
        ]
        metadata_parts.append(
            x.sort_values("date")
            .groupby("permno", as_index=False)
            .tail(1)[metadata_columns]
        )

    if not weekly_parts:
        raise ValueError("No CRSP daily rows were supplied.")

    weekly = pd.concat(weekly_parts, ignore_index=True)
    if weekly.duplicated(["week", "asset"]).any():
        raise ValueError(
            "A weekly return was split or duplicated across chunks. "
            "Use iter_crsp_ciz_daily_chunks so intermediate chunks end on Fridays."
        )
    if cfg.drop_incomplete_final_week and last_observation is not None:
        weekly = weekly.loc[weekly["week"] <= last_observation].copy()

    if cfg.maximum_beta is not None:
        weekly = _apply_lagged_beta_filter(
            weekly,
            float(cfg.maximum_beta),
            int(cfg.beta_lookback_weeks),
            int(cfg.beta_minimum_observations),
        )

    if cfg.top_n_by_market_cap is not None:
        weekly, selected_assets = _rank_lagged_monthly_top_n(
            weekly,
            int(cfg.top_n_by_market_cap),
            int(cfg.ranking_lag_months),
        )
        if not selected_assets:
            raise ValueError(
                "The lagged top-N universe selected no securities. Extend the "
                "download warm-up so minimum_history_days can be reached before "
                "the final formation month, or use --top-n 0 for diagnostics."
            )
        weekly = weekly.loc[weekly["asset"].isin(selected_assets)].copy()
    else:
        weekly["pit_eligible"] = weekly["eligible"]

    returns = weekly.pivot(index="week", columns="asset", values="ret")
    pit = weekly.pivot(index="week", columns="asset", values="pit_eligible")
    returns = returns.sort_index().sort_index(axis=1)
    pit = (
        pit.reindex(index=returns.index, columns=returns.columns)
        .fillna(False)
        .astype(int)
    )
    returns.index.name = "date"
    pit.index.name = "date"

    metadata_all = pd.concat(metadata_parts, ignore_index=True)
    metadata_columns = [
        column
        for column in (
            "permno",
            "ticker",
            "company_name",
            "ues_industry",
            "icb_industry",
            "sic_code",
            "naics_code",
            "primary_exchange",
            "trading_status",
        )
        if column in metadata_all.columns
    ]
    metadata = (
        metadata_all.sort_values("date").groupby("permno", as_index=False).tail(1)
    )
    if cfg.top_n_by_market_cap is not None:
        selected_permnos = {int(asset) for asset in returns.columns}
        metadata = metadata.loc[metadata["permno"].isin(selected_permnos)]
    metadata = metadata[metadata_columns].sort_values("permno").reset_index(drop=True)

    eligible_counts = pit.sum(axis=1)
    diagnostics = {
        "source": "WRDS CRSP CIZ",
        "return_field": "DlyRet",
        "delisting_return_already_included": True,
        "wrds_library": source_attrs.get("wrds_library"),
        "wrds_table": source_attrs.get("wrds_table"),
        "wrds_security_info_table": source_attrs.get("wrds_security_info_table"),
        "n_daily_rows": int(total_daily_rows),
        "n_chunks": int(chunk_count),
        "n_cache_hits": int(cache_hits),
        "n_weeks": len(returns),
        "n_assets": int(returns.shape[1]),
        "minimum_weekly_eligible_assets": int(eligible_counts.min()) if len(pit) else 0,
        "median_weekly_eligible_assets": float(eligible_counts.median())
        if len(pit)
        else 0.0,
        "maximum_weekly_eligible_assets": int(eligible_counts.max()) if len(pit) else 0,
        "first_week": str(returns.index.min().date()) if len(returns) else None,
        "last_week": str(returns.index.max().date()) if len(returns) else None,
        "weekly_config": asdict(cfg),
    }
    return CRSPWeeklyBundle(returns, pit, metadata, diagnostics)


def save_crsp_weekly_bundle(bundle: CRSPWeeklyBundle, output_dir: str | Path) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    bundle.returns.to_csv(destination / "weekly_returns.csv")
    bundle.pit_universe.to_csv(destination / "pit_universe.csv")
    bundle.metadata.to_csv(destination / "crsp_security_metadata.csv", index=False)
    (destination / "wrds_data_manifest.json").write_text(
        json.dumps(bundle.diagnostics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
