from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from akm_hrp.allocators.retail_edge_ml_mpc import (
    RetailEdgeMLMPCAllocator,
    RetailEdgeMLMPCConfig,
    _ml_design,
)
from akm_hrp.allocators.retail_edge_mpc import RetailEdgeMPCConfig
from tests.test_retail_edge_mpc import _edge_fixture


def _ml_allocator() -> tuple[pd.DataFrame, RetailEdgeMLMPCAllocator]:
    returns, pit, features, parent = _edge_fixture()
    features = features.copy()
    features["formation_date"] = returns.index[0]
    assets = returns.columns
    sectors = pd.DataFrame(
        {
            "permno": assets,
            "sec_info_start": "2000-01-01",
            "sec_info_end": "2030-01-01",
            "sector": [f"S{i % 5}" for i in range(len(assets))],
        }
    )
    allocator = RetailEdgeMLMPCAllocator(
        pit,
        structural_features=features,
        sector_history=sectors,
        config=replace(
            RetailEdgeMLMPCConfig(),
            minimum_history_weeks=52,
            risk_lookback_weeks=104,
            maximum_added_assets=6,
            max_added_per_sector=2,
            max_weight=0.10,
            max_sector_weight=0.40,
            max_absolute_style_exposure=1.0,
            weekly_cvar_95_limit=0.30,
            portfolio_value=100_000.0,
            planning_horizon=1,
            optimizer_max_iterations=100,
        ),
    )
    return returns, allocator


def test_ml_design_is_fixed_and_finite() -> None:
    frame = pd.DataFrame(
        np.arange(15, dtype=float).reshape(3, 5),
        columns=[
            "residual_momentum",
            "neglected_earnings_drift",
            "patient_quality_value",
            "conservative_investment",
            "cost_hurdled_liquidity_supply",
        ],
    )
    design = _ml_design(frame)

    assert design.shape == (3, 10)
    assert np.isfinite(design.to_numpy()).all()
    assert design.max(axis=None) <= 6.0


def test_ml_signal_has_no_prior_and_requires_walk_forward_warmup() -> None:
    returns, allocator = _ml_allocator()

    allocator.allocate(returns)

    assert allocator.ic_priors["ml_interaction_ensemble"] == 0.0
    assert allocator.last_signal_weights["ml_interaction_ensemble"] == 0.0
    assert allocator._ml_training_cross_sections == 0


def test_online_ml_learns_only_after_later_returns_arrive() -> None:
    returns, allocator = _ml_allocator()
    first = returns.iloc[:-4]
    allocator.allocate(first)
    saved_design = allocator._previous_ml_design.copy()

    allocator.allocate(returns)

    assert allocator._ml_training_cross_sections == 1
    assert allocator._previous_ml_date == returns.index[-1]
    assert not allocator._previous_ml_design.equals(saved_design)


def test_ml_allocator_rejects_parent_configuration() -> None:
    _, pit, _, _ = _edge_fixture()
    with pytest.raises(TypeError, match="RetailEdgeMLMPCConfig"):
        RetailEdgeMLMPCAllocator(pit, config=RetailEdgeMPCConfig())
