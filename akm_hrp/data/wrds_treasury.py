from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

CRSP_FIXED_TERM_SERIES = {
    2_000_003: "CRSP_TSY_1Y",
    2_000_005: "CRSP_TSY_5Y",
    2_000_007: "CRSP_TSY_10Y",
    2_000_009: "CRSP_TSY_30Y",
}


@dataclass(frozen=True)
class CRSPTreasuryQueryConfig:
    """WRDS locations used for CRSP daily fixed-term Treasury indexes."""

    library: str | None = None
    preferred_libraries: tuple[str, ...] = (
        "crsp_m_treasuries",
        "crsp_q_treasuries",
        "crsp_a_treasuries",
    )
    preferred_tables: tuple[str, ...] = ("tfz_dly_ft",)


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
    if isinstance(description, (list, tuple)):
        columns = [
            str(item.get("name") or item.get("column_name"))
            if isinstance(item, dict)
            else str(item)
            for item in description
        ]
        return [column for column in columns if column and column != "None"]
    raise TypeError("Unsupported WRDS table description format.")


def resolve_crsp_treasury_table(
    connection: Any,
    config: CRSPTreasuryQueryConfig | None = None,
) -> tuple[str, str, dict[str, str]]:
    """Locate the licensed CRSP daily fixed-term Treasury index table."""

    cfg = config or CRSPTreasuryQueryConfig()
    available_libraries = {
        str(name).lower(): str(name) for name in connection.list_libraries()
    }
    requested = (cfg.library,) if cfg.library else cfg.preferred_libraries
    libraries = [
        available_libraries[name.lower()]
        for name in requested
        if name and name.lower() in available_libraries
    ]
    if not libraries:
        expected = (cfg.library,) if cfg.library else cfg.preferred_libraries
        raise RuntimeError(
            "No licensed CRSP Treasury schema is accessible. Expected one of: "
            + ", ".join(str(name) for name in expected)
        )

    for library in libraries:
        tables = {
            str(name).lower(): str(name)
            for name in connection.list_tables(library=library)
        }
        for preferred in cfg.preferred_tables:
            table = tables.get(preferred.lower())
            if table is None:
                continue
            description = connection.describe_table(library=library, table=table)
            lookup = {name.lower(): name for name in _description_columns(description)}
            return_field = lookup.get("tdretadj") or lookup.get("tdretnua")
            required = {
                "treasnox": lookup.get("treasnox"),
                "date": lookup.get("caldt"),
                "return": return_field,
            }
            missing = [key for key, value in required.items() if value is None]
            if missing:
                raise RuntimeError(
                    f"{library}.{table} is missing Treasury fields: {missing}."
                )
            resolved = {key: str(value) for key, value in required.items()}
            return library, table, resolved
    raise RuntimeError(
        "No TFZ_DLY_FT table is accessible in the licensed CRSP Treasury schemas."
    )


def fetch_crsp_fixed_term_daily(
    connection: Any,
    start: str | date,
    end: str | date,
    *,
    series: dict[int, str] | None = None,
    config: CRSPTreasuryQueryConfig | None = None,
) -> pd.DataFrame:
    """Fetch CRSP fixed-term Treasury daily returns as decimal returns."""

    first = pd.Timestamp(start).normalize()
    last = pd.Timestamp(end).normalize()
    if first > last:
        raise ValueError("start must be on or before end.")

    requested = dict(series or CRSP_FIXED_TERM_SERIES)
    if not requested:
        raise ValueError("At least one CRSP TREASNOX series is required.")
    if len(set(requested.values())) != len(requested):
        raise ValueError("Treasury output names must be unique.")
    library, table, columns = resolve_crsp_treasury_table(connection, config)
    safe_library = _safe_identifier(library)
    safe_table = _safe_identifier(table)
    placeholders: list[str] = []
    params: dict[str, object] = {
        "start": first.date().isoformat(),
        "end": last.date().isoformat(),
    }
    for number, identifier in enumerate(requested):
        if int(identifier) != identifier:
            raise ValueError(f"TREASNOX must be an integer: {identifier!r}")
        parameter = f"series_{number}"
        params[parameter] = int(identifier)
        placeholders.append(f"%({parameter})s")
    query = (
        f"SELECT {_safe_identifier(columns['treasnox'])} AS treasnox, "
        f"{_safe_identifier(columns['date'])} AS date, "
        f"{_safe_identifier(columns['return'])} AS raw_return "
        f"FROM {safe_library}.{safe_table} "
        f"WHERE {_safe_identifier(columns['date'])} BETWEEN %(start)s AND %(end)s "
        f"AND {_safe_identifier(columns['treasnox'])} IN ({', '.join(placeholders)}) "
        f"ORDER BY {_safe_identifier(columns['treasnox'])}, "
        f"{_safe_identifier(columns['date'])}"
    )
    daily = connection.raw_sql(query, params=params, date_cols=["date"])
    if daily.empty:
        raise ValueError("CRSP fixed-term Treasury query returned no rows.")
    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily["date"], errors="raise")
    daily["treasnox"] = pd.to_numeric(daily["treasnox"], errors="raise").astype("int64")
    returned_series = set(daily["treasnox"].unique())
    missing_series = sorted(set(requested).difference(returned_series))
    if missing_series:
        raise ValueError(
            f"WRDS returned no rows for requested TREASNOX values: {missing_series}"
        )
    raw = pd.to_numeric(daily["raw_return"], errors="coerce")
    if str(columns["return"]).lower() == "tdretadj":
        # TDRETADJ is TDRETNUA * 100 and is therefore a percentage.
        raw = raw.mask(raw <= -100.0) / 100.0
    else:
        # TDRETNUA uses -99 as its missing-value sentinel.
        raw = raw.mask(raw <= -1.0)
    daily["ret"] = raw.replace([np.inf, -np.inf], np.nan)
    if (daily["ret"].dropna() < -1.0).any():
        raise ValueError("CRSP Treasury returns below -100% were found.")
    daily["asset"] = daily["treasnox"].map(requested)
    if daily["asset"].isna().any():
        raise ValueError("WRDS returned an unrequested TREASNOX series.")
    if daily.duplicated(["date", "asset"]).any():
        raise ValueError("CRSP Treasury data contains duplicate series-date rows.")
    source_metadata = {
        "wrds_library": library,
        "wrds_table": table,
        "return_field": columns["return"],
    }
    daily.attrs.update(source_metadata)
    result = daily[["date", "treasnox", "asset", "ret"]].sort_values(["date", "asset"])
    result.attrs.update(source_metadata)
    return result


def build_crsp_treasury_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    """Compound complete CRSP daily Treasury series to Friday weeks."""

    required = {"date", "asset", "ret"}
    missing = required.difference(daily.columns)
    if missing:
        raise ValueError(f"Treasury daily data is missing columns: {sorted(missing)}")
    values = daily.copy()
    values["date"] = pd.to_datetime(values["date"], errors="raise")
    values["asset"] = values["asset"].astype(str)
    values["ret"] = pd.to_numeric(values["ret"], errors="coerce")
    if values.empty:
        raise ValueError("Treasury daily data contains no rows.")
    if (values["ret"].dropna() < -1.0).any():
        raise ValueError("Treasury daily returns below -100% were found.")
    panel = values.pivot(index="date", columns="asset", values="ret").sort_index()
    valid = panel.notna()
    interior = (~valid) & valid.cummax() & valid.iloc[::-1].cummax().iloc[::-1]
    affected = interior.any(axis=0)
    if affected.any():
        raise ValueError(
            "CRSP Treasury series contain interior daily gaps: "
            f"{affected[affected].index.tolist()}"
        )
    weekly = (1.0 + panel).resample("W-FRI").prod(min_count=1) - 1.0
    weekly.index.name = "date"
    weekly.attrs.update(getattr(daily, "attrs", {}))
    return weekly
