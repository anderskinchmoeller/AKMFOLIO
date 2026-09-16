"""Regression coverage for infeasible MPC exposures and portfolio cleanup."""

import numpy as np
import pandas as pd
import pytest

from akm_hrp.allocators.retail_alpha_ml_mpc import (
    RetailAlphaMLMPCAllocator,
    RetailAlphaMLMPCConfig,
)


@pytest.fixture
def allocator_case():
    rng = np.random.default_rng(1)
    dates = pd.date_range("2020-01-03", periods=120, freq="W-FRI")
    assets = pd.Index([str(30_000 + n) for n in range(25)])
    returns = pd.DataFrame(
        rng.normal(0.0008, 0.012, size=(len(dates), len(assets))),
        index=dates,
        columns=assets,
    )
    core = assets[:15]
    balanced_pit = pd.DataFrame(0, index=dates, columns=assets, dtype=np.int8)
    balanced_pit.loc[:, core] = 1
    features = pd.DataFrame(
        {
            "formation_date": dates[-2],
            "asset": assets,
            "market_cap_usd": np.geomspace(2e8, 2e10, len(assets)),
            "price": 30.0,
            "dollar_volume_20d": np.geomspace(5e6, 80e6, len(assets)),
            "amihud_20d": np.geomspace(0.02, 0.0002, len(assets)),
            "zero_return_fraction_20d": np.linspace(0.08, 0.0, len(assets)),
            "book_to_market": np.linspace(0.2, 1.1, len(assets)),
            "gross_profitability": np.linspace(0.08, 0.45, len(assets)),
            "return_on_assets": np.linspace(0.01, 0.16, len(assets)),
            "accruals_to_assets": np.linspace(0.08, -0.05, len(assets)),
            "standardized_unexpected_earnings": np.linspace(-1.5, 1.5, len(assets)),
            "fundamental_momentum": np.linspace(-0.08, 0.10, len(assets)),
            "shareholder_carry": np.linspace(-0.02, 0.06, len(assets)),
        }
    )
    # Only 3 sectors represented among the selected names; 3 * 25% = 75% < 100%.
    sectors = pd.DataFrame(
        {
            "permno": assets,
            "sec_info_start": "2000-01-01",
            "sec_info_end": "2030-01-01",
            "sector": [f"SECTOR_{n % 3}" for n in range(len(assets))],
        }
    )
    allocator = RetailAlphaMLMPCAllocator(
        balanced_pit,
        structural_features=features,
        sector_history=sectors,
        config=RetailAlphaMLMPCConfig(
            minimum_history_weeks=52,
            risk_lookback_weeks=78,
            ml_max_total_assets=60,
            maximum_added_assets=8,
            # Keep enough names after the sector-aware selection trim.
            max_added_per_sector=10,
            max_weight=0.10,
            max_sector_weight=0.25,
            max_absolute_style_exposure=1.0,
            weekly_cvar_95_limit=0.30,
            portfolio_value=100_000.0,
            planning_horizon=2,
            optimizer_max_iterations=150,
            short_overlay_enabled=False,
            allow_cvar_floor_relaxation=True,
            allow_exposure_limit_relaxation=True,
            ml_minimum_training_cross_sections=8,
        ),
    )

    return allocator, returns.iloc[:80]


def test_sector_cap_infeasibility_raises_actionable_message(allocator_case):
    allocator, returns = allocator_case
    from dataclasses import replace
    allocator.config = replace(allocator.config, allow_exposure_limit_relaxation=False)
    with pytest.raises(RuntimeError) as excinfo:
        allocator.allocate(returns)

    message = str(excinfo.value)
    assert "max_sector_weight" in message
    assert "0.7500" in message or "0.75" in message


def test_floor_keeps_book_when_remaining_capacity_is_insufficient():
    from akm_hrp.allocators.retail_alpha_ml_mpc import _apply_min_weight_floor

    weights = pd.Series([0.4, 0.4, 0.2], index=["a", "b", "c"])
    actual = _apply_min_weight_floor(
        weights, 0.25, pd.Series(0.4, index=weights.index), pd.Index([])
    )
    pd.testing.assert_series_equal(actual, weights)


def test_eligible_cap_error_does_not_raise_name_error(allocator_case):
    allocator, returns = allocator_case
    with pytest.raises(ValueError, match=r"max_weight=0.1000 across 5 eligible assets"):
        allocator.allocate(returns.iloc[:, :5])


def test_selected_cap_is_checked_after_universe_selection(allocator_case):
    allocator, returns = allocator_case
    allocator.balanced_pit.iloc[:, 5:] = False
    with pytest.raises(ValueError, match=r"max_weight=0.1000 across 5 selected assets"):
        allocator.allocate(returns)


def test_floor_still_prunes_when_remaining_capacity_is_sufficient():
    from akm_hrp.allocators.retail_alpha_ml_mpc import _apply_min_weight_floor

    weights = pd.Series([0.49, 0.49, 0.02], index=["a", "b", "c"])
    actual = _apply_min_weight_floor(
        weights, 0.1, pd.Series(0.5, index=weights.index), pd.Index([])
    )
    np.testing.assert_allclose(actual, [0.5, 0.5, 0.0], atol=1e-10)


def test_cleanup_cannot_replace_feasible_solution_with_sector_breach(
    allocator_case, monkeypatch
):
    from dataclasses import replace
    import akm_hrp.allocators.retail_alpha_ml_mpc as module

    allocator, returns = allocator_case
    allocator.config = replace(allocator.config, max_sector_weight=0.5)
    proposed = []

    def bad_cleanup(weights, *args, **kwargs):
        result = weights * 0.0
        sectors = allocator.sectors.lookup(returns.index[-1], weights.index)
        names = sectors.index[sectors == sectors.iloc[0]]
        result.loc[names] = 1.0 / len(names)
        proposed.append(result)
        return result

    monkeypatch.setattr(module, "_apply_min_weight_floor", bad_cleanup)
    weights = allocator.allocate(returns)
    assert proposed
    sectors = allocator.sectors.lookup(returns.index[-1], weights.index)
    assert weights.groupby(sectors).sum().max() <= 0.5 + 1e-7
    assert weights.max() <= allocator.config.max_weight + 1e-7
    assert weights.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(allocator.last_planned_weights.iloc[0], weights)


@pytest.mark.parametrize("dust_weight", [0.000239, 0.0000001, 0.0005])
def test_ml_dust_exit_matches_window_policy_and_is_accounted_for(allocator_case, dust_weight):
    from dataclasses import replace
    from akm_hrp.backtest.engine import _cap_l1_turnover
    allocator, returns = allocator_case
    allocator.config = replace(allocator.config, max_sector_weight=0.50)
    core = returns.columns[:15]
    dust = returns.columns[-1]
    live = pd.Series(0.0, index=returns.columns)
    live.loc[core] = (1.0 - dust_weight) / len(core)
    live.loc[dust] = dust_weight
    allocator.set_current_weights(live)
    prepared = allocator.prepare_allocation_window(returns, returns.drop(columns=dust), live)
    assert dust not in prepared.columns
    target = allocator.allocate(prepared).reindex(live.index, fill_value=0.0)
    assert target.loc[dust] == 0.0
    assert allocator.last_diagnostics.unrepresentable_exit_weight == 0.0
    executed = _cap_l1_turnover(live, target, None)
    assert executed.sum() == pytest.approx(1.0)
    assert (executed - live).loc[dust] == pytest.approx(-dust_weight)
    assert float((executed - live).abs().sum()) >= 2 * dust_weight - 1e-10
    assert allocator.estimate_execution_cost(live, executed) > 0.0


def test_ml_material_missing_holding_still_raises(allocator_case):
    from dataclasses import replace
    allocator, returns = allocator_case
    allocator.config = replace(allocator.config, max_sector_weight=0.50)
    live = pd.Series(0.0, index=returns.columns)
    live.iloc[:15] = .99 / 15
    live.iloc[-1] = .01
    allocator.set_current_weights(live)
    # A truly absent position cannot be restored from the supplied history.
    absent = returns.drop(columns=returns.columns[-1])
    prepared = allocator.prepare_allocation_window(absent, absent, live)
    with pytest.raises(RuntimeError, match="omitted weight=0.010000"):
        allocator.allocate(prepared)


@pytest.mark.parametrize("portfolio_value", [100_000.0, 100_000_000.0])
def test_capacity_pruning_matches_full_optimizer(allocator_case, monkeypatch, portfolio_value):
    import copy
    from dataclasses import replace
    import akm_hrp.allocators.retail_alpha_ml_mpc as module
    allocator, returns = allocator_case
    allocator.config = replace(allocator.config, max_sector_weight=.50, portfolio_value=portfolio_value)
    reference = copy.deepcopy(allocator)
    optimized = allocator.allocate(returns)
    monkeypatch.setattr(module, "_binding_capacity_rows", lambda lower, upper, previous, capacity, steps:
                        np.ones(2 * len(steps) * len(capacity), dtype=bool))
    original = reference.allocate(returns)
    np.testing.assert_allclose(optimized, original, rtol=0, atol=1e-5)
    assert optimized.sum() == pytest.approx(1.0)
    assert optimized.max() <= allocator.config.max_weight + 1e-7


@pytest.mark.parametrize("sector_cap, style_cap", [(0.25, 1.0), (0.25, 0.01), (0.4, 0.01)])
def test_infeasible_exposures_relax_and_continue(allocator_case, sector_cap, style_cap, caplog, monkeypatch):
    from dataclasses import replace
    allocator, returns = allocator_case
    allocator.config = replace(allocator.config, max_absolute_style_exposure=style_cap, max_sector_weight=sector_cap)
    if sector_cap == 0.4:
        import akm_hrp.allocators.retail_alpha_ml_mpc as module
        original = module._current_exposures

        def joint_exposures(window, snapshot, sectors):
            exposures, alpha = original(window, snapshot, sectors)
            # Tight style balance needs ~50% in sector 0, but its cap is 40%.
            # Neither the sector-count nor asset-count bound detects this.
            exposures["SIZE"] = np.where(sectors == "SECTOR_0", -1.0, 1.0)
            return exposures, alpha

        monkeypatch.setattr(module, "_current_exposures", joint_exposures)

    weights = allocator.allocate(returns)
    assert np.isfinite(weights).all()
    assert weights.sum() == pytest.approx(1.0)
    assert weights.min() >= -1e-7
    assert weights.max() <= allocator.config.max_weight + 1e-7
    diagnostics = allocator.last_diagnostics
    slack = diagnostics.exposure_limit_relaxation
    assert slack > 0
    assert diagnostics.maximum_sector_weight <= allocator.config.max_sector_weight + slack + 1e-7
    assert diagnostics.maximum_absolute_style_exposure <= style_cap + slack + 1e-7
    assert diagnostics.as_dict()["exposure_limit_relaxation"] == slack
    assert "infeasible exposure caps" in caplog.text
    # Exercise the next rebalance, including participation limits.
    allocator.set_current_weights(weights)
    again = allocator.allocate(returns)
    assert again.sum() == pytest.approx(1.0)
    assert allocator.last_diagnostics.maximum_participation_utilization <= 1 + 1e-6


@pytest.mark.parametrize("values", [[], [np.nan, np.nan], [np.inf, -np.inf]])
def test_missing_features_have_neutral_zscores_without_warnings(values):
    import warnings
    from akm_hrp.allocators.dynamic_barra_alpha import _robust_zscore
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        actual = _robust_zscore(pd.Series(values, dtype=float))
    np.testing.assert_array_equal(actual, np.zeros(len(values)))


def test_exposure_relaxation_is_minimal_and_resets(allocator_case, monkeypatch):
    from dataclasses import replace
    import akm_hrp.allocators.retail_alpha_ml_mpc as module

    allocator, returns = allocator_case
    original = module._current_exposures

    def neutral_styles(window, snapshot, sectors):
        exposures, alpha = original(window, snapshot, sectors)
        exposures.loc[:, [c for c in exposures if c != "MARKET" and not c.startswith("IND_")]] = 0.0
        return exposures, alpha

    monkeypatch.setattr(module, "_current_exposures", neutral_styles)
    allocator.config = replace(allocator.config, max_absolute_style_exposure=100.0)
    allocator.allocate(returns)
    # Three sector caps must sum to one: 3 * (0.25 + slack) >= 1.
    assert allocator.last_diagnostics.exposure_limit_relaxation == pytest.approx(1 / 3 - .25, abs=1e-7)
    allocator.config = replace(allocator.config, max_sector_weight=1.0)
    allocator.allocate(returns)
    assert allocator.last_diagnostics.exposure_limit_relaxation == 0.0


def test_exposure_relaxation_is_opt_in():
    assert not RetailAlphaMLMPCConfig().allow_exposure_limit_relaxation
