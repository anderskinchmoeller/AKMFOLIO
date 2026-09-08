"""Coverage for the top-level ``model.py`` ML interaction ensemble allocator.

This file previously had no test coverage at all: nothing in ``tests/``
imported ``model`` (the ``retail_alpha_ml_mpc``-named tests target the larger,
separate ``akm_hrp/allocators/retail_alpha_ml_mpc.py``). These tests exercise
the real causal fast/slow ridge learning loop end to end.
"""

import numpy as np
import pandas as pd
import pytest

from model import (
    RetailAlphaMLMPCAllocator,
    RetailAlphaMLMPCConfig,
    _ML_FEATURES,
    _ml_design,
)


def _synthetic_fixture(n_weeks: int = 160, n_assets: int = 40):
    rng = np.random.default_rng(744)
    dates = pd.date_range("2020-01-03", periods=n_weeks, freq="W-FRI")
    assets = pd.Index([str(20_000 + number) for number in range(n_assets)])
    market = rng.normal(0.001, 0.011, size=(len(dates), 1))
    returns = pd.DataFrame(
        market + rng.normal(0.0, 0.012, size=(len(dates), len(assets))),
        index=dates,
        columns=assets,
    )
    returns.iloc[-100:, 15:24] += np.linspace(0.0002, 0.003, 9)
    core = assets[:20]
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
    allocator = RetailAlphaMLMPCAllocator(
        balanced_pit,
        structural_features=features,
        sector_history=sectors,
        config=RetailAlphaMLMPCConfig(
            minimum_history_weeks=52,
            risk_lookback_weeks=78,
            maximum_added_assets=8,
            max_added_per_sector=3,
            max_weight=0.10,
            max_sector_weight=0.40,
            max_absolute_style_exposure=1.0,
            weekly_cvar_95_limit=0.30,
            portfolio_value=100_000.0,
            planning_horizon=2,
            optimizer_max_iterations=150,
            ml_minimum_training_cross_sections=8,
        ),
    )
    return dates, returns, allocator


def test_ml_design_shape_and_bounds():
    idx = pd.Index(["A", "B", "C"])
    signals = pd.DataFrame(
        {
            "momentum_12_1": [3.0, -1.0, 0.5],
            "post_earnings_drift": [1.0, 2.0, -2.0],
            "quality_value_carry": [0.2, -0.2, 1.0],
            "liquid_reversal": [-1.0, 0.0, 0.5],
            "retail_agility": [0.5, 0.5, -0.5],
        },
        index=idx,
    )
    design = _ml_design(signals)
    assert list(design.columns) == list(_ML_FEATURES)
    assert (design.abs() <= 6.0 + 1e-9).all().all()


def test_allocator_runs_causal_walk_forward_without_error():
    """A long expanding-window walk-forward should never raise, always
    produce a fully-invested, non-negative book, and eventually let the ML
    signal earn nonzero blend weight once it has enough training history."""

    dates, returns, allocator = _synthetic_fixture()
    start = 60
    ml_weight_seen_positive = False
    consensus_seen_positive = False

    for i in range(start, len(dates)):
        window = returns.iloc[: i + 1]
        weights = allocator.allocate(window)

        assert weights.sum() == pytest.approx(1.0, abs=1e-6)
        assert (weights >= -1e-9).all()

        if allocator.last_ml_consensus_fraction > 0:
            consensus_seen_positive = True
        ml_weight = allocator.last_signal_weights.get("ml_interaction_ensemble", 0.0)
        if ml_weight > 0:
            ml_weight_seen_positive = True

    assert allocator._ml_training_cross_sections > 0
    assert consensus_seen_positive, "fast/slow models never agreed on any name"
    assert ml_weight_seen_positive, (
        "ml_interaction_ensemble never earned nonzero blend weight across "
        "100 rebalances once past the minimum training-cross-section warmup"
    )


def test_reset_state_clears_ml_learning_state():
    _, returns, allocator = _synthetic_fixture(n_weeks=90)
    for i in range(60, 80):
        allocator.allocate(returns.iloc[: i + 1])
    assert allocator._ml_training_cross_sections > 0

    allocator.reset_state()

    assert allocator._ml_training_cross_sections == 0
    assert allocator._previous_ml_design is None
    assert allocator.last_ml_consensus_fraction == 0.0
    assert np.array_equal(allocator._ml_fast_gram, np.zeros_like(allocator._ml_fast_gram))


def test_sector_cap_infeasibility_raises_actionable_message():
    """Regression test for a RuntimeError users could hit at runtime:
    ``max_sector_weight`` too tight for the number of sectors actually
    represented in the selected book makes a fully-invested portfolio
    mathematically impossible. Previously this surfaced as a bare
    ``RuntimeError: ... HiGHS Status 8`` with no indication of why. The fix
    (in the shared ``akm_hrp/allocators/retail_alpha_mpc.py`` that
    ``RetailAlphaMLMPCAllocator`` inherits ``allocate()`` from unmodified)
    adds a diagnostic that names the binding cap."""

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
            maximum_added_assets=8,
            max_added_per_sector=3,
            max_weight=0.10,
            max_sector_weight=0.25,
            max_absolute_style_exposure=1.0,
            weekly_cvar_95_limit=0.30,
            portfolio_value=100_000.0,
            planning_horizon=2,
            optimizer_max_iterations=150,
            ml_minimum_training_cross_sections=8,
        ),
    )

    with pytest.raises(RuntimeError) as excinfo:
        allocator.allocate(returns.iloc[:80])

    message = str(excinfo.value)
    assert "max_sector_weight" in message
    assert "0.7500" in message or "0.75" in message
