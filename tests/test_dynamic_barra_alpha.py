import numpy as np
import pandas as pd
import pytest

from akm_hrp.allocators.dynamic_barra_alpha import (
    DynamicBarraAlphaAllocator,
    DynamicBarraAlphaConfig,
    _FeatureStore,
)


def test_dynamic_model_keeps_balanced_core_and_adds_candidates() -> None:
    rng = np.random.default_rng(303)
    dates = pd.date_range("2022-01-07", periods=120, freq="W-FRI")
    assets = pd.Index([str(10_000 + number) for number in range(40)])
    returns = pd.DataFrame(
        rng.normal(0.001, 0.015, size=(len(dates), len(assets))),
        index=dates,
        columns=assets,
    )
    returns.iloc[-52:, 20:30] += np.linspace(0.0005, 0.004, 10)
    core = assets[:20]
    balanced_pit = pd.DataFrame(0, index=dates, columns=assets, dtype=np.int8)
    balanced_pit.loc[:, core] = 1
    features = pd.DataFrame(
        {
            "formation_date": dates[-1],
            "asset": assets,
            "market_cap_usd": np.linspace(1e9, 20e9, len(assets)),
            "price": 25.0,
            "dollar_volume_20d": 20_000_000.0,
            "book_to_market": np.linspace(0.2, 1.2, len(assets)),
            "gross_profitability": np.linspace(0.1, 0.5, len(assets)),
            "return_on_assets": np.linspace(0.02, 0.15, len(assets)),
            "accruals_to_assets": np.linspace(0.10, -0.05, len(assets)),
        }
    )
    sector_history = pd.DataFrame(
        {
            "permno": assets,
            "sec_info_start": "2000-01-01",
            "sec_info_end": "2030-01-01",
            "sector": [f"SECTOR_{number % 5}" for number in range(len(assets))],
        }
    )
    allocator = DynamicBarraAlphaAllocator(
        balanced_pit,
        structural_features=features,
        sector_history=sector_history,
        config=DynamicBarraAlphaConfig(
            maximum_added_assets=8,
            max_added_per_sector=3,
            max_weight=0.10,
            max_sector_weight=0.35,
            max_absolute_style_exposure=0.60,
            weekly_cvar_95_limit=0.20,
        ),
    )

    weights = allocator.allocate(returns)

    assert weights.sum() == pytest.approx(1.0)
    assert (weights >= 0.0).all()
    assert core.difference(weights.index).empty
    assert allocator.last_diagnostics.core_asset_count == len(core)
    assert 0 < allocator.last_diagnostics.added_asset_count <= 8
    assert allocator.last_selected_assets.difference(assets).empty
    assert allocator.last_covariance.shape == (len(weights), len(weights))
    assert allocator.last_diagnostics.maximum_participation_utilization <= 1.0


def test_feature_store_carries_each_pit_field_from_its_own_release() -> None:
    features = pd.DataFrame(
        [
            {
                "formation_date": "2025-01-03",
                "asset": "10001",
                "book_to_market": 0.7,
            },
            {
                "formation_date": "2025-02-07",
                "asset": "10001",
                "market_cap_usd": 2e9,
            },
        ]
    )
    store = _FeatureStore(features)

    snapshot = store.snapshot(pd.Timestamp("2025-02-07"), pd.Index(["10001"]))

    assert snapshot.loc["10001", "book_to_market"] == pytest.approx(0.7)
    assert snapshot.loc["10001", "market_cap_usd"] == pytest.approx(2e9)
