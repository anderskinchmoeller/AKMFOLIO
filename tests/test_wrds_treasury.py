import numpy as np
import pandas as pd
import pytest

from akm_hrp.data.wrds_treasury import (
    build_crsp_treasury_weekly,
    fetch_crsp_fixed_term_daily,
    resolve_crsp_treasury_table,
)


class FakeTreasuryConnection:
    def __init__(self) -> None:
        self.query = ""
        self.params = {}

    def list_libraries(self):
        return ["crsp_m_treasuries"]

    def list_tables(self, library):
        return ["TFZ_DLY_FT"]

    def describe_table(self, library, table):
        return pd.DataFrame({"name": ["TREASNOX", "CALDT", "TDRETADJ"]})

    def raw_sql(self, query, params, date_cols):
        self.query = query
        self.params = params
        rows = []
        for date in pd.bdate_range("2025-01-06", periods=5):
            rows.extend(
                [
                    {"treasnox": 2_000_003, "date": date, "raw_return": 0.10},
                    {"treasnox": 2_000_005, "date": date, "raw_return": 0.20},
                ]
            )
        return pd.DataFrame(rows)


def test_treasury_table_resolution_and_percentage_conversion() -> None:
    connection = FakeTreasuryConnection()
    library, table, columns = resolve_crsp_treasury_table(connection)
    assert library == "crsp_m_treasuries"
    assert table == "TFZ_DLY_FT"
    assert columns["return"] == "TDRETADJ"

    daily = fetch_crsp_fixed_term_daily(
        connection,
        "2025-01-01",
        "2025-01-31",
        series={2_000_003: "CRSP_TSY_1Y", 2_000_005: "CRSP_TSY_5Y"},
    )

    assert daily["ret"].iloc[0] == pytest.approx(0.001)
    assert connection.params["series_0"] == 2_000_003
    assert "tfz_dly_ft" in connection.query.lower()


def test_treasury_weekly_returns_are_geometrically_compounded() -> None:
    connection = FakeTreasuryConnection()
    daily = fetch_crsp_fixed_term_daily(
        connection,
        "2025-01-01",
        "2025-01-31",
        series={2_000_003: "CRSP_TSY_1Y", 2_000_005: "CRSP_TSY_5Y"},
    )

    weekly = build_crsp_treasury_weekly(daily)

    assert weekly.loc["2025-01-10", "CRSP_TSY_1Y"] == pytest.approx(1.001**5 - 1.0)
    assert weekly.loc["2025-01-10", "CRSP_TSY_5Y"] == pytest.approx(1.002**5 - 1.0)
    assert np.isfinite(weekly.to_numpy()).all()


def test_treasury_download_fails_when_a_requested_series_is_absent() -> None:
    with pytest.raises(ValueError, match="2000007"):
        fetch_crsp_fixed_term_daily(
            FakeTreasuryConnection(),
            "2025-01-01",
            "2025-01-31",
            series={
                2_000_003: "CRSP_TSY_1Y",
                2_000_005: "CRSP_TSY_5Y",
                2_000_007: "CRSP_TSY_10Y",
            },
        )
