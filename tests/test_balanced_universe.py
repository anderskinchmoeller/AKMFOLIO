import numpy as np
import pandas as pd
import pytest

from akm_hrp.data.balanced_universe import (
    BalancedUniverseConfig,
    attach_point_in_time_sectors,
    build_style_factor_proxies,
    merge_point_in_time_characteristics,
    monthly_membership_to_weekly_pit,
    select_balanced_monthly_membership,
)


def _features() -> pd.DataFrame:
    rows = []
    sectors = {"10": "Technology", "20": "Healthcare", "30": "Utilities"}
    for date in pd.to_datetime(["2025-01-31", "2025-02-28"]):
        for sector_number, (prefix, sector) in enumerate(sectors.items()):
            for rank in range(4):
                rows.append(
                    {
                        "formation_date": date,
                        "asset": f"{prefix}{rank}",
                        "sector": sector,
                        "market_cap_usd": 1e9 * (rank + 1),
                        "price": 20.0,
                        "dollar_volume_20d": 5e6 - rank,
                    }
                )
    return pd.DataFrame(rows)


def test_point_in_time_sector_join_respects_effective_interval() -> None:
    features = pd.DataFrame(
        {
            "formation_date": pd.to_datetime(["2025-01-31", "2025-03-31"]),
            "asset": [10001.0, 10001.0],
            "market_cap_usd": [1e9, 1e9],
            "price": [20.0, 20.0],
            "dollar_volume_20d": [5e6, 5e6],
        }
    )
    history = pd.DataFrame(
        {
            "permno": [10001, 10001],
            "sec_info_start": pd.to_datetime(["2024-01-01", "2025-03-01"]),
            "sec_info_end": pd.to_datetime(["2025-02-28", "2026-12-31"]),
            "sector": ["Technology", "Industrials"],
        }
    )

    result = attach_point_in_time_sectors(features, history)

    assert result["asset"].tolist() == ["10001", "10001"]
    assert result["sector"].tolist() == ["Technology", "Industrials"]


def test_open_ended_sector_interval_survives_csv_date_limits() -> None:
    features = pd.DataFrame(
        {
            "formation_date": ["2025-01-31"],
            "asset": [10001],
            "market_cap_usd": [1e9],
            "price": [20.0],
            "dollar_volume_20d": [5e6],
        }
    )
    history = pd.DataFrame(
        {
            "permno": [10001],
            "sec_info_start": ["2020-01-01"],
            "sec_info_end": ["9999-12-31"],
            "sector": ["Technology"],
        }
    )

    result = attach_point_in_time_sectors(features, history)

    assert result.loc[0, "sector"] == "Technology"


def test_noavail_sector_falls_back_to_historic_sic_branch() -> None:
    """Older CRSP rows lack UES/ICB labels but retain their SIC code."""

    features = pd.DataFrame(
        {
            "formation_date": ["2001-01-31"],
            "asset": [10001],
            "market_cap_usd": [1e9],
            "price": [20.0],
            "dollar_volume_20d": [5e6],
        }
    )
    history = pd.DataFrame(
        {
            "permno": [10001],
            "sec_info_start": ["1999-01-01"],
            "sec_info_end": ["2002-01-01"],
            "sector": ["NOAVAIL"],
            "sic_code": [3571],
        }
    )

    result = attach_point_in_time_sectors(features, history)

    assert result.loc[0, "sector"] == "SIC_35"


def test_overlapping_sector_intervals_fail_closed() -> None:
    features = pd.DataFrame(
        {
            "formation_date": ["2025-01-31"],
            "asset": [10001],
            "market_cap_usd": [1e9],
            "price": [20.0],
            "dollar_volume_20d": [5e6],
        }
    )
    history = pd.DataFrame(
        {
            "permno": [10001, 10001],
            "sec_info_start": ["2020-01-01", "2024-01-01"],
            "sec_info_end": ["2025-01-31", "2026-01-31"],
            "sector": ["Technology", "Industrials"],
        }
    )

    with pytest.raises(ValueError, match="overlapping effective intervals"):
        attach_point_in_time_sectors(features, history)


def test_balanced_selection_caps_each_sector_and_lags_membership() -> None:
    membership = select_balanced_monthly_membership(
        _features(),
        BalancedUniverseConfig(
            stocks_per_sector=2,
            minimum_size_percentile=0.0,
            minimum_sectors=3,
            selection_lag_months=1,
        ),
    )

    per_sector = membership.groupby(["formation_month", "sector"]).size()
    assert (per_sector == 2).all()
    assert membership["effective_month"].min() == pd.Period("2025-02", freq="M")

    dates = pd.date_range("2025-01-03", "2025-03-28", freq="W-FRI")
    returns = pd.DataFrame(
        0.0, index=dates, columns=sorted(_features()["asset"].unique())
    )
    safety = pd.DataFrame(1, index=dates, columns=returns.columns)
    safety.loc[:, "100"] = 0
    pit = monthly_membership_to_weekly_pit(membership, returns, base_pit=safety)

    assert (pit.loc["2025-01"].sum(axis=1) == 0).all()
    assert (pit.loc["2025-02"].sum(axis=1) == 5).all()
    assert (pit.loc["2025-03"].sum(axis=1) == 5).all()


def test_balanced_config_rejects_negative_liquidity_floor() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        select_balanced_monthly_membership(
            _features(),
            BalancedUniverseConfig(minimum_median_dollar_volume=-1.0),
        )


def test_release_aware_value_characteristic_requires_exact_weekly_key() -> None:
    structural = _features().drop(columns="sector").iloc[:2].copy()
    values = pd.DataFrame(
        {
            "formation_date": structural["formation_date"],
            "permno": structural["asset"].astype(float),
            "book_to_market": [0.4, 0.8],
        }
    )

    merged = merge_point_in_time_characteristics(structural, values)

    assert merged["book_to_market"].tolist() == pytest.approx([0.4, 0.8])


def test_style_factors_use_prior_month_rankings_and_emit_value() -> None:
    assets = [str(10000 + number) for number in range(8)]
    dates = pd.date_range("2025-01-03", "2025-02-28", freq="W-FRI")
    realized_by_asset = np.linspace(0.001, 0.008, len(assets))
    returns = pd.DataFrame(
        np.tile(realized_by_asset, (len(dates), 1)),
        index=dates,
        columns=assets,
    )
    rows = []
    for month_end, reverse in [
        (pd.Timestamp("2025-01-31"), False),
        (pd.Timestamp("2025-02-28"), True),
    ]:
        order = np.arange(1, 9, dtype=float)
        if reverse:
            order = order[::-1]
        for position, asset in enumerate(assets):
            rows.append(
                {
                    "formation_date": month_end,
                    "asset": asset,
                    "market_cap_usd": order[position] * 1e9,
                    "book_to_market": order[::-1][position] / 10.0,
                    "price": 20.0,
                    "dollar_volume_20d": 5e6,
                }
            )
    features = pd.DataFrame(rows)
    factors = build_style_factor_proxies(
        returns,
        features,
        eligible_mask=pd.DataFrame(1, index=dates, columns=assets),
        minimum_leg_assets=2,
        quantile=0.25,
    )

    february = factors.loc["2025-02"]
    assert "CRSP_VALUE" in factors
    # January's small/value rankings are held during February even though the
    # February characteristic snapshot reverses them.
    assert (february["CRSP_SIZE"] < 0.0).all()
    assert (february["CRSP_VALUE"] < 0.0).all()
