import numpy as np
import pandas as pd

from akm_hrp.data.wrds_crsp import (
    CRSPChunkConfig,
    CRSPQueryConfig,
    CRSPWeeklyConfig,
    build_crsp_weekly_bundle,
    build_crsp_weekly_bundle_from_chunks,
    fetch_crsp_ciz_daily,
    fetch_crsp_sector_history,
    iter_crsp_ciz_daily_chunks,
    resolve_crsp_table,
)


def test_sector_history_prefers_ues_then_icb_then_sic() -> None:
    class FakeConnection:
        query = ""

        def list_libraries(self):
            return ["crsp"]

        def list_tables(self, library):
            return ["stkDlySecurityData", "stkSecurityInfoHist"]

        def describe_table(self, library, table):
            if table == "stkDlySecurityData":
                names = ["PERMNO", "DlyCalDt", "DlyRet"]
            else:
                names = [
                    "PERMNO",
                    "SecInfoStartDt",
                    "SecInfoEndDt",
                    "UESIndustry",
                    "ICBIndustry",
                    "SICCD",
                    "TradingTicker",
                ]
            return pd.DataFrame({"name": names})

        def raw_sql(self, query, params, date_cols):
            self.query = query
            assert params == {"start": "2020-01-01", "end": "2020-12-31"}
            return pd.DataFrame(
                {
                    "permno": [10001, 10002, 10003],
                    "sec_info_start": pd.to_datetime(["2019-01-01"] * 3),
                    "sec_info_end": pd.to_datetime(["2021-12-31"] * 3),
                    "ues_industry": ["Technology", None, None],
                    "icb_industry": ["Software", "Utilities", None],
                    "sic_code": ["7372", "4911", "3571"],
                    "ticker": ["AAA", "BBB", "CCC"],
                }
            )

    connection = FakeConnection()
    result = fetch_crsp_sector_history(
        connection,
        "2020-01-01",
        "2020-12-31",
    )

    assert result["sector"].tolist() == ["Technology", "Utilities", "SIC_35"]
    assert "secinfoenddt >= %(start)s" in " ".join(connection.query.lower().split())


def test_ciz_returns_are_compounded_once_and_keyed_by_permno():
    rows = []
    dates = pd.date_range("2020-01-06", periods=10, freq="B")
    for permno in (10101, 20202):
        for day in dates:
            rows.append(
                {
                    "permno": permno,
                    "date": day,
                    "ret": 0.01,
                    "price": 20.0,
                    "volume": 1000.0,
                    "ticker": f"T{permno}",
                }
            )

    bundle = build_crsp_weekly_bundle(
        pd.DataFrame(rows),
        CRSPWeeklyConfig(
            minimum_history_days=1,
            minimum_price=5.0,
            drop_incomplete_final_week=False,
        ),
    )

    expected = (1.01**5) - 1.0
    assert list(bundle.returns.columns) == ["10101", "20202"]
    assert np.allclose(bundle.returns.to_numpy(), expected)
    assert (bundle.pit_universe == 1).all().all()
    assert bundle.diagnostics["delisting_return_already_included"] is True


def test_delisting_loss_is_not_multiplied_twice():
    daily = pd.DataFrame(
        {
            "permno": [10101, 10101],
            "date": pd.to_datetime(["2020-01-06", "2020-01-07"]),
            "ret": [0.10, -1.0],
            "price": [10.0, 0.0],
        }
    )
    bundle = build_crsp_weekly_bundle(
        daily,
        CRSPWeeklyConfig(
            minimum_history_days=1,
            minimum_price=None,
            drop_incomplete_final_week=False,
        ),
    )
    assert bundle.returns.iloc[0, 0] == -1.0


def test_schema_discovery_prefers_current_crsp_access():
    class FakeConnection:
        def list_libraries(self):
            return ["crsp_a_stock", "crsp_a_ccm", "crsp"]

        def list_tables(self, library):
            if library == "crsp":
                return ["stkDlySecurityData"]
            return ["dsf_v2"]

        def describe_table(self, library, table):
            return pd.DataFrame({"name": ["PERMNO", "DlyCalDt", "DlyRet", "DlyPrc"]})

    library, table, columns = resolve_crsp_table(
        FakeConnection(),
        CRSPQueryConfig(),
    )
    assert library == "crsp"
    assert table == "stkDlySecurityData"
    assert columns["ret"] == "DlyRet"


def test_raw_ciz_daily_uses_date_effective_security_classification():
    class FakeConnection:
        query = ""

        def list_libraries(self):
            return ["crsp", "crsp_a_stock"]

        def list_tables(self, library):
            assert library == "crsp"
            return ["stkDlySecurityData", "stkSecurityInfoHist"]

        def describe_table(self, library, table):
            if table == "stkDlySecurityData":
                names = ["PERMNO", "DlyCalDt", "DlyRet", "DlyPrc", "DlyVol"]
            else:
                names = [
                    "PERMNO",
                    "SecInfoStartDt",
                    "SecInfoEndDt",
                    "TradingTicker",
                    "IssuerName",
                    "ShareType",
                    "SecurityType",
                    "SecuritySubType",
                    "USIncFlg",
                    "IssuerType",
                ]
            return pd.DataFrame({"name": names})

        def raw_sql(self, query, params, date_cols):
            self.query = query
            assert params == {"start": "2020-01-01", "end": "2020-01-31"}
            assert date_cols == ["date"]
            return pd.DataFrame(
                {
                    "permno": [10101],
                    "date": pd.to_datetime(["2020-01-02"]),
                    "ret": [0.01],
                    "price": [10.0],
                    "volume": [100.0],
                    "ticker": ["TEST"],
                    "company_name": ["Test Corp"],
                    "share_type": ["NS"],
                    "security_type": ["EQTY"],
                    "security_subtype": ["COM"],
                    "us_incorporated": ["Y"],
                    "issuer_type": ["CORP"],
                }
            )

    connection = FakeConnection()
    result = fetch_crsp_ciz_daily(
        connection,
        "2020-01-01",
        "2020-01-31",
    )
    normalized_query = " ".join(connection.query.lower().split())
    assert "join crsp.stksecurityinfohist i" in normalized_query
    assert "d.dlycaldt between i.secinfostartdt and i.secinfoenddt" in normalized_query
    assert "i.sharetype = 'ns'" in normalized_query
    assert "i.issuertype in ('acor', 'corp')" in normalized_query
    assert result.attrs["wrds_security_info_table"] == "stkSecurityInfoHist"


def test_raw_ciz_daily_supports_explicit_etf_ticker_filters():
    class FakeConnection:
        def __init__(self):
            self.query = ""
            self.params = {}

        def list_libraries(self):
            return ["crsp"]

        def list_tables(self, library):
            return ["dsf_v2"]

        def describe_table(self, library, table):
            return pd.DataFrame(
                {
                    "name": [
                        "PERMNO",
                        "DlyCalDt",
                        "DlyRet",
                        "DlyPrc",
                        "DlyVol",
                        "Ticker",
                        "ShareType",
                        "SecurityType",
                        "SecuritySubType",
                        "USIncFlg",
                        "IssuerType",
                    ]
                }
            )

        def raw_sql(self, query, params, date_cols):
            self.query = query
            self.params = params
            return pd.DataFrame(
                {
                    "permno": [84398],
                    "date": pd.to_datetime(["2020-01-02"]),
                    "ret": [0.01],
                    "price": [320.0],
                    "volume": [1_000_000.0],
                    "ticker": ["SPY"],
                    "share_type": ["NS"],
                    "security_type": ["FUND"],
                    "security_subtype": ["ETF"],
                    "us_incorporated": ["Y"],
                    "issuer_type": ["ACOR"],
                }
            )

    connection = FakeConnection()
    result = fetch_crsp_ciz_daily(
        connection,
        "2020-01-01",
        "2020-01-31",
        CRSPQueryConfig(
            common_stocks_only=False,
            tickers=("SPY", "GLD"),
            share_types=("NS",),
            security_types=("FUND",),
            security_subtypes=("ETF", "ETV"),
        ),
    )

    normalized_query = " ".join(connection.query.lower().split())
    assert "d.securitytype in (%(filter_security_type_0)s)" in normalized_query
    assert "d.securitysubtype in" in normalized_query
    assert "upper(d.ticker) in" in normalized_query
    assert connection.params["filter_ticker_0"] == "SPY"
    assert connection.params["filter_ticker_1"] == "GLD"
    assert connection.params["filter_security_type_0"] == "FUND"
    assert result.loc[0, "ticker"] == "SPY"


def test_monthly_top_n_is_formed_with_a_one_month_lag():
    caps = {
        "2020-01": {10101: 300.0, 20202: 200.0, 30303: 100.0},
        "2020-02": {10101: 100.0, 20202: 200.0, 30303: 300.0},
        "2020-03": {10101: 100.0, 20202: 200.0, 30303: 300.0},
    }
    rows = []
    for day in pd.date_range("2020-01-03", "2020-03-27", freq="W-FRI"):
        month_caps = caps[str(day.to_period("M"))]
        for permno, market_cap in month_caps.items():
            rows.append(
                {
                    "permno": permno,
                    "date": day,
                    "ret": 0.001,
                    "price": 20.0,
                    "market_cap": market_cap,
                }
            )

    bundle = build_crsp_weekly_bundle(
        pd.DataFrame(rows),
        CRSPWeeklyConfig(
            minimum_history_days=1,
            minimum_price=5.0,
            top_n_by_market_cap=2,
            ranking_lag_months=1,
            drop_incomplete_final_week=False,
        ),
    )

    january = bundle.pit_universe.loc["2020-01"]
    february = bundle.pit_universe.loc["2020-02"]
    march = bundle.pit_universe.loc["2020-03"]
    assert (january.sum(axis=1) == 0).all()
    assert (february[["10101", "20202"]] == 1).all().all()
    assert (february["30303"] == 0).all()
    assert (march[["20202", "30303"]] == 1).all().all()
    assert (march["10101"] == 0).all()


def test_low_beta_filter_uses_only_lagged_returns() -> None:
    rows = []
    dates = pd.date_range("2020-01-03", periods=80, freq="W-FRI")
    market = 0.01 * np.sin(np.arange(len(dates)) / 3.0)
    for permno, beta in ((10101, 0.4), (20202, 1.0), (30303, 1.6)):
        for position, day in enumerate(dates):
            rows.append(
                {
                    "permno": permno,
                    "date": day,
                    "ret": beta * market[position],
                    "price": 20.0,
                    "market_cap": 100.0,
                }
            )

    bundle = build_crsp_weekly_bundle(
        pd.DataFrame(rows),
        CRSPWeeklyConfig(
            minimum_history_days=1,
            minimum_price=1.0,
            maximum_beta=0.8,
            beta_lookback_weeks=26,
            beta_minimum_observations=12,
            drop_incomplete_final_week=False,
        ),
    )

    latest = bundle.pit_universe.iloc[-1]
    assert latest["10101"] == 1
    assert latest["20202"] == 0
    assert latest["30303"] == 0


def test_maximum_market_cap_creates_an_actual_microcap_ceiling() -> None:
    rows = []
    for permno, market_cap in ((10101, 50_000_000.0), (20202, 500_000_000.0)):
        for day in pd.date_range("2020-01-06", periods=5, freq="B"):
            rows.append(
                {
                    "permno": permno,
                    "date": day,
                    "ret": 0.001,
                    "price": 10.0,
                    "market_cap": market_cap,
                }
            )

    bundle = build_crsp_weekly_bundle(
        pd.DataFrame(rows),
        CRSPWeeklyConfig(
            minimum_history_days=1,
            minimum_price=1.0,
            minimum_market_cap=20_000_000.0,
            maximum_market_cap=300_000_000.0,
            drop_incomplete_final_week=False,
        ),
    )

    assert bundle.pit_universe.iloc[-1]["10101"] == 1
    assert bundle.pit_universe.iloc[-1]["20202"] == 0


def test_chunked_weekly_aggregation_preserves_history_across_chunks():
    rows = []
    for day in pd.date_range("2020-01-03", "2020-02-28", freq="W-FRI"):
        for permno in (10101, 20202):
            rows.append(
                {
                    "permno": permno,
                    "date": day,
                    "ret": 0.01,
                    "price": 20.0,
                    "market_cap": float(permno),
                }
            )
    daily = pd.DataFrame(rows)
    chunks = [
        daily.loc[daily["date"] <= "2020-01-31"].copy(),
        daily.loc[daily["date"] > "2020-01-31"].copy(),
    ]
    config = CRSPWeeklyConfig(
        minimum_history_days=2,
        minimum_price=5.0,
        drop_incomplete_final_week=False,
    )

    expected = build_crsp_weekly_bundle(daily, config)
    actual = build_crsp_weekly_bundle_from_chunks(chunks, config)

    pd.testing.assert_frame_equal(actual.returns, expected.returns)
    pd.testing.assert_frame_equal(actual.pit_universe, expected.pit_universe)
    assert actual.diagnostics["n_chunks"] == 2


def test_chunk_cache_resumes_without_requerying_wrds(tmp_path):
    class FakeConnection:
        def __init__(self):
            self.raw_calls = 0

        def list_libraries(self):
            return ["crsp"]

        def list_tables(self, library):
            return ["dsf_v2"]

        def describe_table(self, library, table):
            return pd.DataFrame(
                {
                    "name": [
                        "PERMNO",
                        "DlyCalDt",
                        "DlyRet",
                        "DlyPrc",
                        "DlyCap",
                        "ShareType",
                        "SecurityType",
                        "SecuritySubType",
                        "USIncFlg",
                        "IssuerType",
                    ]
                }
            )

        def raw_sql(self, query, params, date_cols):
            self.raw_calls += 1
            return pd.DataFrame(
                {
                    "permno": [10101],
                    "date": pd.to_datetime([params["start"]]),
                    "ret": [0.01],
                    "price": [20.0],
                    "market_cap": [100.0],
                    "share_type": ["NS"],
                    "security_type": ["EQTY"],
                    "security_subtype": ["COM"],
                    "us_incorporated": ["Y"],
                    "issuer_type": ["CORP"],
                }
            )

    config = CRSPChunkConfig(
        months_per_chunk=1,
        cache_dir=tmp_path,
        resume=True,
    )
    first_connection = FakeConnection()
    first = list(
        iter_crsp_ciz_daily_chunks(
            first_connection,
            "2020-01-01",
            "2020-03-31",
            chunk_config=config,
        )
    )
    assert first_connection.raw_calls == len(first)
    assert first_connection.raw_calls > 1

    resumed_connection = FakeConnection()
    resumed = list(
        iter_crsp_ciz_daily_chunks(
            resumed_connection,
            "2020-01-01",
            "2020-03-31",
            chunk_config=config,
        )
    )
    assert resumed_connection.raw_calls == 0
    assert all(chunk.attrs["chunk_cache_hit"] for chunk in resumed)
