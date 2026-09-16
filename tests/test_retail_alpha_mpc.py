from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from akm_hrp.allocators.retail_alpha_mpc import (
    RetailAlphaMPCAllocator,
    RetailAlphaMPCConfig,
    _SIGNALS,
    _capped_normalize,
    _impact_cost_arrays,
    _retail_signal_panel,
)
from akm_hrp.allocators.dynamic_barra_alpha import _current_exposures
from akm_hrp.backtest.engine import run_walk_forward
from akm_hrp.config import HRPConfig


def _model_fixture() -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, RetailAlphaMPCAllocator
]:
    rng = np.random.default_rng(744)
    dates = pd.date_range("2023-01-06", periods=82, freq="W-FRI")
    assets = pd.Index([str(20_000 + number) for number in range(30)])
    market = rng.normal(0.001, 0.011, size=(len(dates), 1))
    returns = pd.DataFrame(
        market + rng.normal(0.0, 0.012, size=(len(dates), len(assets))),
        index=dates,
        columns=assets,
    )
    returns.iloc[-52:, 15:24] += np.linspace(0.0002, 0.003, 9)
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
    sectors = pd.DataFrame(
        {
            "permno": assets,
            "sec_info_start": "2000-01-01",
            "sec_info_end": "2030-01-01",
            "sector": [f"SECTOR_{number % 5}" for number in range(len(assets))],
        }
    )
    allocator = RetailAlphaMPCAllocator(
        balanced_pit,
        structural_features=features,
        sector_history=sectors,
        config=RetailAlphaMPCConfig(
            minimum_history_weeks=52,
            risk_lookback_weeks=78,
            maximum_added_assets=6,
            max_added_per_sector=2,
            max_weight=0.10,
            max_sector_weight=0.40,
            max_absolute_style_exposure=1.0,
            weekly_cvar_95_limit=0.30,
            portfolio_value=100_000.0,
            planning_horizon=2,
            optimizer_max_iterations=150,
        ),
    )
    return returns, balanced_pit, features, allocator


def test_retail_alpha_mpc_keeps_core_forecasts_risk_and_plans_path() -> None:
    returns, balanced_pit, _, allocator = _model_fixture()
    weights = allocator.allocate(returns.iloc[:-1])

    core = balanced_pit.columns[:15]
    assert weights.sum() == pytest.approx(1.0)
    assert (weights >= 0.0).all()
    assert core.difference(weights.index).empty
    assert 0 < allocator.last_diagnostics.added_asset_count <= 6
    assert allocator.last_planned_weights.shape == (2, len(weights))
    assert np.isfinite(allocator.last_specific_volatility).all()
    assert (allocator.last_specific_volatility > 0.0).all()
    assert allocator.last_signal_weights.sum() == pytest.approx(1.0)
    assert allocator.last_signal_weights.max() <= 0.40 + 1e-10
    assert set(allocator.last_diagnostics.as_dict()).issuperset(
        {
            "estimated_temporary_impact",
            "estimated_permanent_impact",
            "signal_weight__post_earnings_drift",
            "signal_rank_ic__momentum_12_1",
        }
    )
    assert allocator.config.min_weight == 0.0


def test_dynamic_ic_updates_only_after_forward_returns_arrive() -> None:
    returns, _, _, allocator = _model_fixture()
    allocator.allocate(returns.iloc[:-1])
    before = allocator.last_diagnostics.signal_ic_observations.copy()

    allocator.allocate(returns)
    after = allocator.last_diagnostics.signal_ic_observations

    assert all(after[name] >= before[name] for name in before)
    assert any(after[name] > before[name] for name in before)


def test_temporary_and_permanent_impact_are_nonlinear_in_trade_size() -> None:
    config = RetailAlphaMPCConfig(portfolio_value=1_000_000.0)
    volatility = np.array([0.04])
    adv = np.array([2_000_000.0])
    spread = np.array([0.0004])

    small = _impact_cost_arrays(
        np.array([0.01]), volatility, adv, spread, config
    )
    large = _impact_cost_arrays(
        np.array([0.02]), volatility, adv, spread, config
    )

    assert large[0] == pytest.approx(2.0 * small[0])
    assert large[1] > 2.0 * small[1]
    assert large[2] == pytest.approx(4.0 * small[2])


def test_backtest_uses_allocator_specific_execution_cost() -> None:
    class CostAwareAllocator:
        class Config:
            minimum_history_weeks = 2
            requires_pairwise_finite_correlation = True
            max_weight = 1.0

        config = Config()

        def set_current_weights(self, weights: pd.Series) -> None:
            self.current = weights

        def allocate(self, returns: pd.DataFrame) -> pd.Series:
            return pd.Series([0.6, 0.4], index=returns.columns)

        def estimate_execution_cost(
            self, previous: pd.Series, target: pd.Series
        ) -> float:
            return 0.001 if float((target - previous).abs().sum()) > 1e-12 else 0.0

    dates = pd.date_range("2025-01-03", periods=8, freq="W-FRI")
    returns = pd.DataFrame(
        {"A": np.linspace(0.00, 0.02, 8), "B": np.linspace(0.01, -0.01, 8)},
        index=dates,
    )
    config = replace(
        HRPConfig(),
        lookback_weeks=8,
        min_window_obs=2,
        min_holding_weeks=1,
        drift_threshold=0.0,
        max_rebalance_turnover_l1=None,
        tc_bps=0.0,
    )

    result = run_walk_forward(returns, CostAwareAllocator(), config)

    assert result.transaction_costs is not None
    assert result.transaction_costs.sum() > 0.0


def test_balanced_only_sleeve_is_restored_to_allocation_window() -> None:
    returns, balanced_pit, _, allocator = _model_fixture()
    sleeve = pd.Series(
        np.linspace(-0.01, 0.015, len(returns)), index=returns.index
    )
    full = returns.assign(BALANCED_SLEEVE=sleeve)
    expanded_pit = balanced_pit.assign(BALANCED_SLEEVE=1)
    restored_allocator = RetailAlphaMPCAllocator(
        expanded_pit,
        config=allocator.config,
    )

    prepared = restored_allocator.prepare_allocation_window(
        full, returns.copy(), None
    )

    assert "BALANCED_SLEEVE" in prepared


def test_infeasible_optimizer_result_is_rejected() -> None:
    returns, _, _, allocator = _model_fixture()
    allocator.config = replace(
        allocator.config,
        optimizer_max_iterations=1,
        weekly_cvar_95_limit=1e-6,
    )

    with pytest.raises(RuntimeError, match="constraint-feasible"):
        allocator.allocate(returns.iloc[:-1])


def test_capped_normalize_never_allocates_to_unavailable_signals() -> None:
    confidence = pd.Series(
        [1.0, 0.0, 0.0, 0.0, 0.0], index=pd.Index(_SIGNALS)
    )

    weights = _capped_normalize(confidence, cap=0.40)

    assert weights.iloc[0] == pytest.approx(1.0)
    assert (weights.iloc[1:] == 0.0).all()


def test_signal_cleaner_reports_coverage_and_restores_exact_neutrality() -> None:
    returns, _, features, allocator = _model_fixture()
    window = returns.iloc[:-1]
    snapshot = features.set_index("asset").reindex(window.columns)
    sectors = allocator.sectors.lookup(window.index[-1], window.columns)
    exposures, _ = _current_exposures(window, snapshot, sectors)
    caps = snapshot["market_cap_usd"]

    signals, coverage, masks = _retail_signal_panel(
        window, snapshot, exposures, caps
    )

    neutral_columns = [
        column
        for column in exposures
        if column == "MARKET" or column.startswith("IND_")
    ]
    weighted_moments = (
        exposures[neutral_columns].mul(caps / caps.sum(), axis=0).T @ signals
    )
    assert weighted_moments.abs().to_numpy().max() < 1e-10
    assert coverage.index.tolist() == list(_SIGNALS)
    assert coverage.between(0.0, 1.0).all()
    assert masks.shape == signals.shape


def test_sparse_feature_signal_and_sub_noise_floor_ic_are_turned_off() -> None:
    returns, _, features, allocator = _model_fixture()
    window = returns.iloc[:-1]
    snapshot = features.set_index("asset").reindex(window.columns)
    snapshot.loc[
        snapshot.index[5:],
        ["standardized_unexpected_earnings", "fundamental_momentum"],
    ] = np.nan
    sectors = allocator.sectors.lookup(window.index[-1], window.columns)
    exposures, _ = _current_exposures(window, snapshot, sectors)
    signals, coverage, masks = _retail_signal_panel(
        window, snapshot, exposures, snapshot["market_cap_usd"]
    )
    availability = (
        signals.std(ddof=0).gt(1e-8)
        & coverage.ge(allocator.config.minimum_signal_coverage)
    ).astype(float)
    assert availability["post_earnings_drift"] == 0.0
    assert (~masks.loc[snapshot.index[5:], "post_earnings_drift"]).all()
    assert (
        signals.loc[snapshot.index[5:], "post_earnings_drift"] == 0.0
    ).all()

    allocator.config = replace(allocator.config, signal_ic_noise_floor=0.015)
    _, weights = allocator._signal_combination(
        pd.Series(1.0, index=pd.Index(_SIGNALS)), signals
    )
    assert weights["quality_value_carry"] == 0.0
    assert weights["liquid_reversal"] == 0.0
    assert weights["retail_agility"] == 0.0
    assert weights.sum() == pytest.approx(1.0)


def test_no_candidates_are_added_when_every_cleaned_signal_is_off() -> None:
    returns, balanced_pit, _, allocator = _model_fixture()
    allocator.config = replace(allocator.config, signal_ic_noise_floor=0.05)

    allocator.allocate(returns.iloc[:-1])

    assert allocator.last_signal_weights.sum() == pytest.approx(0.0)
    assert allocator.last_selected_assets.difference(
        balanced_pit.columns[:15]
    ).empty
    assert allocator.last_diagnostics.added_asset_count == 0


def test_dynamic_ic_uses_previous_assets_even_after_current_exit() -> None:
    returns, _, _, allocator = _model_fixture()
    assets = returns.columns[:10]
    previous_date, current_date = returns.index[-2:]
    scores = pd.Series(np.arange(len(assets), dtype=float), index=assets)
    allocator._previous_signal_panel = pd.DataFrame(
        {signal: scores for signal in _SIGNALS}
    )
    allocator._previous_signal_availability = pd.DataFrame(
        True, index=assets, columns=_SIGNALS
    )
    allocator._previous_signal_date = previous_date
    current_survivors = returns.loc[[previous_date, current_date], assets[:7]]
    allocator._engine_return_context = returns.loc[
        [previous_date, current_date], assets
    ]

    allocator._update_dynamic_ics(current_survivors, current_date)

    assert (allocator._ic_weight == 1.0).all()


def test_dynamic_ic_excludes_assets_without_formation_signal() -> None:
    returns, _, _, allocator = _model_fixture()
    assets = returns.columns[:10]
    previous_date, current_date = returns.index[-2:]
    scores = pd.Series(np.arange(len(assets), dtype=float), index=assets)
    allocator._previous_signal_panel = pd.DataFrame(
        {signal: scores for signal in _SIGNALS}
    )
    masks = pd.DataFrame(True, index=assets, columns=_SIGNALS)
    masks.loc[assets[2:], "post_earnings_drift"] = False
    allocator._previous_signal_availability = masks
    allocator._previous_signal_date = previous_date
    allocator._engine_return_context = returns.loc[
        [previous_date, current_date], assets
    ]

    allocator._update_dynamic_ics(
        allocator._engine_return_context, current_date
    )

    assert allocator._ic_weight["post_earnings_drift"] == 0.0
    assert allocator._ic_weight.drop("post_earnings_drift").eq(1.0).all()


def test_exiting_holding_is_costed_and_liquidated_within_capacity() -> None:
    returns, balanced_pit, _, allocator = _model_fixture()
    full = returns.iloc[:-1]
    allocator.allocate(full)
    core = balanced_pit.columns[:15]
    added = allocator.last_selected_assets.difference(core)[0]
    live = pd.Series(0.0, index=returns.columns)
    live.loc[core[:10]] = 0.095
    live.loc[added] = 0.05
    allocator.set_current_weights(live)
    prepared = allocator.prepare_allocation_window(
        full, full.drop(columns=[added]), live
    )

    target = allocator.allocate(prepared)

    assert added in allocator.last_selected_assets
    assert target.loc[added] == pytest.approx(0.0, abs=1e-9)
    assert allocator.last_diagnostics.target_turnover_l1 >= 0.05 - 1e-9


def test_sparse_live_holding_is_repaired_for_costed_exit() -> None:
    returns, balanced_pit, _, allocator = _model_fixture()
    full = returns.iloc[:-1].copy()
    allocator.allocate(full)
    core = balanced_pit.columns[:15]
    added = allocator.last_selected_assets.difference(core)[0]
    live = pd.Series(0.0, index=returns.columns)
    live.loc[core[:10]] = 0.095
    live.loc[added] = 0.05
    sparse = full.copy()
    sparse.loc[sparse.index[:-8], added] = np.nan
    eligible = sparse.drop(columns=[added])
    allocator.set_current_weights(live)

    prepared = allocator.prepare_allocation_window(sparse, eligible, live)
    target = allocator.allocate(prepared)

    assert added in prepared
    assert prepared[added].notna().all()
    assert added in allocator.last_selected_assets
    assert target.loc[added] == pytest.approx(0.0, abs=1e-9)
    assert allocator.last_diagnostics.unrepresentable_exit_weight == 0.0


def test_dust_holding_below_min_weight_is_not_force_carried() -> None:
    """A leftover position below the configured min_weight should be written
    off (dropped from `held`/forced exits) instead of being ramped forever,
    per the exit_materiality_threshold fix: prepare_allocation_window reuses
    min_weight (falling back to a small fixed floor when min_weight == 0)
    instead of only excluding floating-point noise."""
    returns, balanced_pit, _, allocator = _model_fixture()
    full = returns.iloc[:-1]
    allocator.allocate(full)
    core = balanced_pit.columns[:15]
    added = allocator.last_selected_assets.difference(core)[0]

    allocator.config = replace(allocator.config, min_weight=0.02)

    live = pd.Series(0.0, index=returns.columns)
    live.loc[core[:10]] = 0.095
    live.loc[added] = 0.01  # below min_weight=0.02, above floating-point _EPS
    allocator.set_current_weights(live)

    prepared = allocator.prepare_allocation_window(
        full, full.drop(columns=[added]), live
    )

    assert added not in allocator._forced_exit_assets
    assert added not in prepared.columns


@pytest.mark.parametrize(
    "dust_weight",
    [
        0.000157,  # first reported instance (2026-09-09 full-scale run)
        0.000243,  # second reported instance, different rebalance date
    ],
)
def test_dust_holding_below_materiality_floor_does_not_abort_allocate(
    dust_weight: float,
) -> None:
    """Regression test for a real crash found running the model at full
    broad-universe scale: a leftover position sized between _EPS and the
    5e-4 materiality floor (e.g. weight=1.57e-4, from turnover-cap drift)
    is correctly judged immaterial by prepare_allocation_window and left
    off the window (see test_dust_holding_below_min_weight_is_not_force_
    carried above) -- but the subsequent allocate() call still summed that
    same dust weight as an "omitted" live holding and compared it against
    its own, much tighter maximum_unrepresentable_weight (default 1e-8),
    raising "RuntimeError: Live holdings could not be represented in the
    MPC window; omitted weight=<value>." and aborting the entire
    walk-forward run over a position the model had already decided to
    write off. Fixed by having allocate() reuse the same materiality floor
    (self._held_materiality_threshold, set by prepare_allocation_window) to
    decide what counts as a real omission instead of a write-off.

    Parametrized over every distinct dust weight actually reported in the
    wild so far -- the fix is a threshold comparison, not a special case for
    one literal number, and a second live run surfacing a second value
    (0.000243, at a different rebalance date) is exactly the scenario this
    parametrization exists to keep covered."""
    returns, balanced_pit, _, allocator = _model_fixture()
    full = returns.iloc[:-1]
    allocator.allocate(full)
    core = balanced_pit.columns[:15]
    added = allocator.last_selected_assets.difference(core)[0]

    live = pd.Series(0.0, index=returns.columns)
    live.loc[core[:10]] = 0.095
    live.loc[added] = dust_weight  # dust: above _EPS, below the 5e-4 floor
    allocator.set_current_weights(live)

    prepared = allocator.prepare_allocation_window(
        full, full.drop(columns=[added]), live
    )
    assert added not in prepared.columns  # written off, as before the fix

    target = allocator.allocate(prepared)  # must not raise

    assert added not in target.index
    assert allocator.last_diagnostics.unrepresentable_exit_weight == pytest.approx(
        0.0, abs=1e-9
    )


def test_material_holding_at_or_above_min_weight_is_still_force_carried() -> None:
    """The companion case: a position that is still above the materiality
    floor must keep being restored as a forced exit, same as before the
    fix -- the threshold should only write off genuine dust."""
    returns, balanced_pit, _, allocator = _model_fixture()
    full = returns.iloc[:-1]
    allocator.allocate(full)
    core = balanced_pit.columns[:15]
    added = allocator.last_selected_assets.difference(core)[0]

    allocator.config = replace(allocator.config, min_weight=0.02)

    live = pd.Series(0.0, index=returns.columns)
    live.loc[core[:10]] = 0.095
    live.loc[added] = 0.03  # above min_weight=0.02
    allocator.set_current_weights(live)

    prepared = allocator.prepare_allocation_window(
        full, full.drop(columns=[added]), live
    )

    assert added in allocator._forced_exit_assets
    assert added in prepared.columns
