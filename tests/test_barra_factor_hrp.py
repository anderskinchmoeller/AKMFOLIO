from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from akm_hrp.allocators.barra_factor_hrp import (
    BarraFactorHRPAllocator,
    BarraFactorHRPConfig,
    barra_factor_covariance,
)
from akm_hrp.backtest.engine import latest_target_weights, run_walk_forward
from akm_hrp.config import HRPConfig


def test_factor_covariance_and_hrp_weights_are_finite() -> None:
    rng = np.random.default_rng(11)
    returns = pd.DataFrame(
        rng.normal(0.0, 0.02, size=(80, 12)),
        columns=[str(asset) for asset in range(12)],
    )

    covariance, diagnostics = barra_factor_covariance(returns)
    allocator = BarraFactorHRPAllocator(BarraFactorHRPConfig(max_weight=0.10))
    weights = allocator.allocate(returns)

    assert np.isfinite(covariance.to_numpy()).all()
    assert diagnostics.factor_count >= 1
    assert weights.sum() == pytest.approx(1.0)
    assert (weights >= 0.0).all()
    assert weights.max() <= 0.10 + 1e-12
    assert allocator.last_diagnostics.effective_asset_count >= 10.0
    covariance = allocator.last_covariance.loc[weights.index, weights.index]
    contribution = weights.to_numpy() * (covariance.to_numpy() @ weights.to_numpy())
    expected_maximum = contribution.max() / contribution.sum()
    assert allocator.last_diagnostics.maximum_risk_contribution == pytest.approx(
        expected_maximum
    )


def test_staggered_external_factor_history_drops_sparse_columns() -> None:
    rng = np.random.default_rng(23)
    dates = pd.date_range("2000-01-07", periods=40, freq="W-FRI")
    returns = pd.DataFrame(
        rng.normal(0.0, 0.02, size=(40, 12)),
        index=dates,
        columns=[str(asset) for asset in range(12)],
    )
    factors = pd.DataFrame(
        {
            "SIZE": rng.normal(0.0, 0.01, size=40),
            "VALUE_STARTS_LATE": [np.nan] * 30
            + rng.normal(0.0, 0.01, size=10).tolist(),
        },
        index=dates,
    )

    _, diagnostics = barra_factor_covariance(
        returns,
        factor_returns=factors,
    )

    assert diagnostics.factor_source == "external"
    assert diagnostics.factor_count == 1


def test_external_factor_gap_uses_causal_pca_fallback() -> None:
    rng = np.random.default_rng(29)
    dates = pd.date_range("1990-01-05", periods=30, freq="W-FRI")
    returns = pd.DataFrame(
        rng.normal(0.0, 0.02, size=(30, 12)),
        index=dates,
        columns=[str(asset) for asset in range(12)],
    )
    unavailable_factors = pd.DataFrame(
        np.nan,
        index=dates,
        columns=["SIZE", "VALUE", "MOMENTUM"],
    )

    _, diagnostics = barra_factor_covariance(
        returns,
        factor_returns=unavailable_factors,
    )

    assert diagnostics.factor_source == "statistical_pca"
    assert diagnostics.factor_count >= 1



def test_broad_universe_is_not_silently_infeasible() -> None:
    """Regression test for a real bug found running the model against the
    full ~2,400-name universe: BarraFactorHRPConfig had no min_weight field,
    so engine.py's _portfolio_bounds_feasible() fell back to HRPConfig's
    default min_weight=0.01 (tuned for concentrated <=100-name books).
    n_eligible * 0.01 > 1 for any universe above ~100 names, so
    run_walk_forward silently skipped every single rebalance (0 completed,
    no error at all) and latest_target_weights hard-crashed with
    "Latest target bounds are infeasible". Fixed by giving
    BarraFactorHRPConfig its own min_weight=0.0, matching RAHRPConfig.
    """
    rng = np.random.default_rng(7)
    n_assets = 150
    n_weeks = 90
    returns = pd.DataFrame(
        rng.normal(0.0, 0.02, size=(n_weeks, n_assets)),
        index=pd.date_range("2020-01-03", periods=n_weeks, freq="W-FRI"),
        columns=[str(asset) for asset in range(n_assets)],
    )

    allocator = BarraFactorHRPAllocator(BarraFactorHRPConfig())
    assert allocator.config.min_weight == 0.0

    cfg = replace(
        HRPConfig(),
        lookback_weeks=52,
        min_window_obs=26,
        min_holding_weeks=0,
        drift_threshold=0.0,
        max_rebalance_turnover_l1=None,
        tc_bps=0.0,
        cov_max_interior_missing_fraction=1.0,
    )
    assert cfg.min_weight == 0.01  # the shared default that caused the bug

    result = run_walk_forward(returns, allocator, cfg, progress_every_rebalances=0)
    assert result.weights.abs().sum(axis=1).gt(0).sum() > 0, (
        "every rebalance was silently skipped -- min_weight bounds check "
        "regressed"
    )

    target = latest_target_weights(returns, allocator, cfg)
    assert target.sum() == pytest.approx(1.0)
    assert (target >= 0.0).all()
