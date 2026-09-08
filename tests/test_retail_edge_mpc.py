import numpy as np
import pandas as pd
import pytest
from dataclasses import replace

from akm_hrp.allocators.dynamic_barra_alpha import _current_exposures
from akm_hrp.allocators.retail_edge_mpc import (
    RetailEdgeMPCAllocator,
    RetailEdgeMPCConfig,
    capacity_edge_signal_panel,
)
from akm_hrp.allocators.retail_alpha_mpc import RetailAlphaMPCConfig
from akm_hrp.diagnostics.significance import (
    newey_west_mean_test,
    sharpe_significance,
)


def _edge_fixture():
    rng = np.random.default_rng(902)
    dates = pd.date_range("2021-01-08", periods=120, freq="W-FRI")
    assets = pd.Index([str(30_000 + i) for i in range(30)])
    core = assets[:15]
    common = rng.normal(0.001, 0.010, size=(len(dates), 1))
    values = common + rng.normal(0.0, 0.014, size=(len(dates), len(assets)))
    values[-52:-4, 15:24] += np.linspace(0.0005, 0.0030, 9)
    returns = pd.DataFrame(values, index=dates, columns=assets)
    pit = pd.DataFrame(0, index=dates, columns=assets, dtype=np.int8)
    pit.loc[:, core] = 1
    features = pd.DataFrame(
        {
            "formation_date": dates[-1],
            "asset": assets,
            "market_cap_usd": np.r_[
                np.full(15, 5_000_000_000.0),
                np.geomspace(50_000_000.0, 900_000_000.0, 15),
            ],
            "price": 20.0,
            "dollar_volume_20d": 5_000_000.0,
            "turnover_20d": 0.01,
            "amihud_20d": 0.001,
            "zero_return_fraction_20d": 0.01,
            "standardized_unexpected_earnings": np.linspace(-1.0, 2.0, 30),
            "fundamental_momentum": np.linspace(-0.05, 0.12, 30),
            "book_to_market": np.linspace(0.2, 1.3, 30),
            "gross_profitability": np.linspace(0.05, 0.50, 30),
            "return_on_assets": np.linspace(0.01, 0.18, 30),
            "cash_return_on_assets": np.linspace(0.00, 0.20, 30),
            "accruals_to_assets": np.linspace(0.10, -0.08, 30),
            "asset_growth": np.linspace(0.30, -0.10, 30),
            "sales_growth": np.linspace(0.25, -0.05, 30),
        }
    )
    sectors = pd.DataFrame(
        {
            "permno": assets,
            "sec_info_start": "2000-01-01",
            "sec_info_end": "2030-01-01",
            "sector": [f"S{i % 5}" for i in range(30)],
        }
    )
    config = RetailEdgeMPCConfig(
        minimum_history_weeks=52,
        risk_lookback_weeks=104,
        maximum_added_assets=6,
        max_added_per_sector=2,
        max_weight=0.10,
        max_sector_weight=0.40,
        max_absolute_style_exposure=1.0,
        weekly_cvar_95_limit=0.30,
        portfolio_value=100_000.0,
        planning_horizon=2,
        optimizer_max_iterations=150,
    )
    allocator = RetailEdgeMPCAllocator(
        pit,
        structural_features=features,
        sector_history=sectors,
        config=config,
    )
    return returns, pit, features, allocator


def test_capacity_gate_excludes_large_core_but_keeps_retail_niche() -> None:
    returns, pit, features, allocator = _edge_fixture()
    snapshot = features.set_index("asset").reindex(returns.columns)
    sectors = allocator.sectors.lookup(returns.index[-1], returns.columns)
    exposures, _ = _current_exposures(returns, snapshot, sectors)

    signals, coverage, masks = capacity_edge_signal_panel(
        returns,
        snapshot,
        exposures,
        snapshot["market_cap_usd"],
        returns,
        allocator.config,
    )

    assert not masks.loc[pit.columns[:15]].any(axis=None)
    assert masks.loc[pit.columns[15:]].any(axis=None)
    assert (signals.loc[~masks.any(axis=1)] == 0.0).all(axis=None)
    assert coverage["residual_momentum"] == pytest.approx(0.5)


def test_retail_edge_allocator_adds_only_capacity_eligible_assets() -> None:
    returns, pit, _, allocator = _edge_fixture()

    weights = allocator.allocate(returns)
    additions = allocator.last_selected_assets.difference(pit.columns[:15])

    assert weights.sum() == pytest.approx(1.0)
    assert 0 < len(additions) <= 6
    assert allocator.last_signal_weights.index.tolist() == list(
        allocator.signal_names
    )
    assert allocator.last_diagnostics.added_asset_count == len(additions)


def test_deflated_sharpe_penalizes_multiple_search_trials() -> None:
    rng = np.random.default_rng(12)
    returns = pd.Series(rng.normal(0.002, 0.01, 260))

    single = sharpe_significance(returns, number_of_trials=1)
    searched = sharpe_significance(returns, number_of_trials=50)

    assert searched["deflated_sharpe_benchmark"] > 0.0
    assert searched["deflated_sharpe_ratio"] < single["deflated_sharpe_ratio"]
    assert searched["significance_trials"] == 50.0


def test_newey_west_test_reports_active_alpha_statistics() -> None:
    active = pd.Series(np.tile([0.003, 0.001, 0.002, 0.004], 65))

    result = newey_west_mean_test(active)

    assert result["annualized_mean"] > 0.0
    assert result["hac_t_stat"] > 2.0
    assert result["hac_p_value"] < 0.05


def test_edge_config_rejects_parent_config() -> None:
    _, pit, _, _ = _edge_fixture()
    with pytest.raises(TypeError, match="RetailEdgeMPCConfig"):
        RetailEdgeMPCAllocator(pit, config=RetailAlphaMPCConfig())


def test_unattainable_cvar_uses_disclosed_minimum_risk_floor() -> None:
    returns, _, _, allocator = _edge_fixture()
    allocator.config = replace(
        allocator.config,
        weekly_cvar_95_limit=1e-6,
        optimizer_max_iterations=1,
    )

    weights = allocator.allocate(returns)

    assert weights.sum() == pytest.approx(1.0)
    assert allocator.last_diagnostics.optimizer_repaired
    assert allocator.last_diagnostics.risk_limit_relaxed
    assert allocator.last_diagnostics.effective_weekly_cvar_limit > 1e-6
