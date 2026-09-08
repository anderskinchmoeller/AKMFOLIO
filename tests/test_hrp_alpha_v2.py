import numpy as np
import pandas as pd
import pytest

from akm_hrp.allocators.hrp_alpha_v2 import (
    HRPAlphaV2Allocator,
    HRPAlphaV2Config,
    causal_ridge_score,
    combine_structural_feature_tables,
    structural_anomaly_score,
    structural_snapshot,
)
from akm_hrp.cli.build_structural_alpha_features import build_structural_features
from build_compustat_pit_features import (
    BuildConfig,
    align_to_weekly_universe,
    prepare_quarterly_features,
)


def _returns(observations: int = 260, assets: int = 20, seed: int = 31) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2018-01-05", periods=observations, freq="W-FRI")
    factors = rng.normal(
        [0.001, 0.0005, 0.0], [0.020, 0.015, 0.010], size=(observations, 3)
    )
    loadings = rng.normal(0.0, 0.6, size=(3, assets))
    values = factors @ loadings + rng.normal(0.0, 0.018, size=(observations, assets))
    return pd.DataFrame(
        values, index=index, columns=[str(10000 + i) for i in range(assets)]
    )


def _features(returns: pd.DataFrame) -> pd.DataFrame:
    count = returns.shape[1]
    return pd.DataFrame(
        {
            "formation_date": returns.index[-1],
            "asset": returns.columns,
            "market_cap_usd": np.geomspace(100_000_000.0, 20_000_000_000.0, count),
            "dollar_volume_20d": np.geomspace(200_000.0, 20_000_000.0, count),
            "gross_profitability": np.linspace(-0.2, 0.4, count),
            "leverage": np.linspace(0.8, 0.1, count),
        }
    )


def test_v2_combines_systematic_ml_and_structural_scores() -> None:
    returns = _returns()
    allocator = HRPAlphaV2Allocator(
        HRPAlphaV2Config(max_weight=0.15, portfolio_value=10_000.0),
        structural_features=_features(returns),
    )

    weights = allocator.allocate(returns)

    assert weights.sum() == pytest.approx(1.0, abs=1e-9)
    assert weights.min() >= -1e-12
    assert weights.max() <= 0.15 + 1e-9
    assert allocator.last_diagnostics is not None
    assert allocator.last_diagnostics.tree_count == 3
    assert allocator.last_diagnostics.ml_training_dates >= 16
    assert 0.0 <= allocator.last_diagnostics.ml_confidence <= 1.0
    assert allocator.last_diagnostics.structural_feature_coverage == pytest.approx(1.0)
    assert len(allocator.last_diagnostics.ml_coefficients) == 5


def test_structural_snapshot_cannot_use_future_rows() -> None:
    returns = _returns()
    assets = returns.columns
    as_of = returns.index[-1]
    past = _features(returns)
    future = past.copy()
    future["formation_date"] = as_of + pd.Timedelta(weeks=1)
    future["gross_profitability"] = 999.0
    combined = pd.concat([past, future], ignore_index=True)

    snapshot = structural_snapshot(combined, as_of, assets, max_age_days=45)

    assert snapshot["gross_profitability"].max() < 1.0


def test_microcap_liquidity_capacity_and_event_caps_are_enforced() -> None:
    returns = _returns(assets=10)
    features = _features(returns)
    features.loc[0, "market_cap_usd"] = 100_000_000.0
    features.loc[1, "dollar_volume_20d"] = 50_000.0
    features["merger_spread"] = np.nan
    features.loc[2, "merger_spread"] = 0.10
    config = HRPAlphaV2Config(
        max_weight=0.40,
        microcap_max_weight=0.03,
        event_max_weight=0.04,
        minimum_dollar_volume=100_000.0,
        portfolio_value=10_000.0,
    )
    allocator = HRPAlphaV2Allocator(config, structural_features=features)

    weights = allocator.allocate(returns)

    assert weights.iloc[0] <= 0.03 + 1e-9
    assert weights.iloc[1] == pytest.approx(0.0, abs=1e-9)
    assert weights.iloc[2] <= 0.04 + 1e-9
    assert allocator.last_diagnostics is not None
    assert allocator.last_diagnostics.microcap_asset_count >= 1
    assert allocator.last_diagnostics.liquidity_exclusion_count == 1
    assert allocator.last_diagnostics.event_signal_asset_count == 1


def test_ridge_model_is_chronological_and_confidence_gated() -> None:
    returns = _returns()
    config = HRPAlphaV2Config()

    score, dates, validation_ic, confidence, coefficients = causal_ridge_score(
        returns,
        config,
    )

    assert score.index.equals(returns.columns)
    assert np.isfinite(score).all()
    assert dates >= config.ml_min_training_dates
    assert np.isfinite(validation_ic)
    assert 0.0 <= confidence <= 1.0
    assert len(coefficients) == 5


def test_multiple_structural_tables_merge_sparse_fields() -> None:
    date = pd.Timestamp("2025-12-26")
    market = pd.DataFrame({"date": [date], "asset": ["10001"], "market_cap_usd": [2e8]})
    fundamental = pd.DataFrame(
        {
            "formation_date": [date],
            "permno": ["10001"],
            "gross_profitability": [0.3],
        }
    )

    combined = combine_structural_feature_tables([market, fundamental])

    assert combined is not None
    assert combined.loc[0, "market_cap_usd"] == pytest.approx(2e8)
    assert combined.loc[0, "gross_profitability"] == pytest.approx(0.3)


def test_event_scores_can_be_derived_from_point_in_time_terms() -> None:
    snapshot = pd.DataFrame(
        {
            "current_price": [95.0, 48.0, 9.0],
            "offer_price": [100.0, np.nan, np.nan],
            "deal_success_probability": [0.9, np.nan, np.nan],
            "deal_break_price": [70.0, np.nan, np.nan],
            "cef_nav": [np.nan, 55.0, np.nan],
            "warrant_fair_value": [np.nan, np.nan, 12.0],
        },
        index=["DEAL", "CEF", "WARRANT"],
    )

    score, event_signal = structural_anomaly_score(snapshot)

    assert event_signal.notna().all()
    assert score.notna().all()


def test_daily_cache_builder_lags_weekly_microstructure_features(tmp_path) -> None:
    dates = pd.bdate_range("2025-01-02", periods=35)
    rows = []
    for asset, scale in [("10001", 1.0), ("10002", 2.0)]:
        for position, date in enumerate(dates):
            rows.append(
                {
                    "permno": asset,
                    "date": date,
                    "ret": 0.001 * scale,
                    "price": 10.0 * scale,
                    "market_cap": 200_000.0 * scale,
                    "volume": 10_000.0 + position,
                }
            )
    cache = pd.DataFrame(rows)
    path = tmp_path / "crsp_dsf_v2_20250101_20250301_common1.csv.gz"
    cache.to_csv(path, index=False)

    features = build_structural_features(tmp_path, rolling_days=10)

    assert {"market_cap_usd", "dollar_volume_20d", "amihud_20d"}.issubset(
        features.columns
    )
    assert features["asset"].nunique() == 2
    assert features["formation_date"].is_monotonic_increasing
    assert features["market_cap_usd"].dropna().min() >= 200_000_000.0


def test_compustat_builder_uses_lagged_crsp_cap_for_book_to_market(tmp_path) -> None:
    dates = pd.date_range("2019-03-31", periods=8, freq="QE")
    fundamentals = pd.DataFrame(
        {
            "gvkey": "001000",
            "permno": 10001,
            "datadate": dates,
            "rdq": dates + pd.Timedelta(days=40),
            "fyearq": dates.year,
            "fqtr": np.tile([1, 2, 3, 4], 2),
            "atq": 100.0,
            "ltq": 40.0,
            "ceqq": 60.0,
            "saleq": 30.0,
            "cogsq": 15.0,
            "ibq": 3.0,
            "actq": 50.0,
            "lctq": 20.0,
            "dlcq": 5.0,
            "dlttq": 20.0,
            "oancfy": np.tile([3.0, 7.0, 12.0, 18.0], 2),
            "xrdq": 1.0,
        }
    )
    config = BuildConfig(maximum_staleness_days=180)
    quarterly, _ = prepare_quarterly_features(fundamentals, {"10001"}, config)
    weekly_index = pd.date_range("2020-01-03", "2021-06-25", freq="W-FRI")
    pit = pd.DataFrame(1, index=weekly_index, columns=["10001"])
    # Raw CRSP DlyCap is in thousands; 100,000 becomes USD 100 million.
    market_caps = pd.DataFrame(100_000.0, index=weekly_index, columns=["10001"])
    destination = tmp_path / "features.csv.gz"

    align_to_weekly_universe(
        quarterly,
        weekly_index,
        pit,
        destination,
        config,
        market_caps,
    )
    aligned = pd.read_csv(destination)

    assert aligned["book_to_market"].dropna().median() == pytest.approx(0.60)


def test_reset_state_clears_v2_path_data() -> None:
    returns = _returns()
    allocator = HRPAlphaV2Allocator(
        HRPAlphaV2Config(max_weight=0.15, portfolio_value=10_000.0),
        structural_features=_features(returns),
    )
    allocator.allocate(returns)

    allocator.reset_state()

    assert allocator.last_diagnostics is None
    assert allocator.last_composite_score is None
    assert allocator._previous_weights is None
