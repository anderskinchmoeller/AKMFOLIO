import numpy as np
import pandas as pd
import pytest

from akm_hrp.allocators.hrp_alpha_v1 import (
    HRPAlphaV1Allocator,
    HRPAlphaV1Config,
    blended_ledoit_wolf_covariance,
    momentum_and_trend_scores,
)
from akm_hrp.backtest.engine import _compute_metrics


def _sample(
    observations: int = 130,
    assets: int = 12,
    seed: int = 21,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2020-01-03", periods=observations, freq="W-FRI")
    market = rng.normal(0.001, 0.020, size=(observations, 1))
    loadings = rng.uniform(0.3, 1.2, size=(1, assets))
    residual = rng.normal(0.0, 0.018, size=(observations, assets))
    values = market @ loadings + residual
    return pd.DataFrame(
        values,
        index=index,
        columns=[f"asset_{i}" for i in range(assets)],
    )


def test_allocator_returns_bounded_ensemble_hrp_weights() -> None:
    returns = _sample()
    allocator = HRPAlphaV1Allocator(HRPAlphaV1Config(max_weight=0.20))

    weights = allocator.allocate(returns)

    assert weights.index.equals(returns.columns)
    assert np.isfinite(weights).all()
    assert weights.sum() == pytest.approx(1.0, abs=1e-9)
    assert weights.min() >= -1e-12
    assert weights.max() <= 0.20 + 1e-9
    assert allocator.last_diagnostics is not None
    assert allocator.last_diagnostics.tree_count == 3
    assert allocator.last_diagnostics.long_observations == 104
    assert allocator.last_diagnostics.short_observations == 26


def test_blended_ledoit_wolf_covariance_is_positive_definite() -> None:
    returns = _sample()
    covariance, long_shrinkage, short_shrinkage, long_obs, short_obs = (
        blended_ledoit_wolf_covariance(returns, HRPAlphaV1Config())
    )

    eigenvalues = np.linalg.eigvalsh(covariance.to_numpy(dtype=float))
    assert eigenvalues.min() > 0.0
    assert 0.0 <= long_shrinkage <= 1.0
    assert 0.0 <= short_shrinkage <= 1.0
    assert (long_obs, short_obs) == (104, 26)


def test_momentum_tilt_overweights_the_strongest_asset() -> None:
    returns = _sample(seed=7)
    # Give one asset a persistent medium-term premium large enough to dominate
    # its rank while preserving non-zero covariance with the other assets.
    returns.loc[returns.index[-70:], "asset_0"] += 0.012
    neutral_config = HRPAlphaV1Config(
        max_weight=0.40,
        momentum_tilt_strength=0.0,
        negative_trend_multiplier=1.0,
    )
    tilted_config = HRPAlphaV1Config(
        max_weight=0.40,
        momentum_tilt_strength=0.75,
        negative_trend_multiplier=1.0,
    )

    neutral = HRPAlphaV1Allocator(neutral_config).allocate(returns)
    tilted = HRPAlphaV1Allocator(tilted_config).allocate(returns)
    score, trend = momentum_and_trend_scores(returns, tilted_config)

    assert score.idxmax() == "asset_0"
    assert trend["asset_0"] > 0.0
    assert tilted["asset_0"] > neutral["asset_0"]


def test_no_trade_band_freezes_small_target_changes() -> None:
    returns = _sample()
    allocator = HRPAlphaV1Allocator(
        HRPAlphaV1Config(max_weight=0.20, no_trade_band=1.0)
    )
    first = allocator.allocate(returns)

    second = allocator.allocate(returns.copy())

    pd.testing.assert_series_equal(first, second, atol=1e-12, rtol=1e-12)
    assert allocator.last_diagnostics is not None
    assert allocator.last_diagnostics.frozen_by_no_trade_band == len(first)
    assert allocator.last_diagnostics.target_turnover_l1 == pytest.approx(0.0)


def test_volatility_target_requires_and_uses_real_cash_asset() -> None:
    returns = _sample(assets=6)
    rng = np.random.default_rng(3)
    returns["BIL"] = rng.normal(0.0007, 0.0001, size=len(returns))
    allocator = HRPAlphaV1Allocator(
        HRPAlphaV1Config(
            max_weight=0.40,
            annual_target_volatility=0.04,
            cash_asset="BIL",
            max_cash_weight=0.80,
        )
    )

    weights = allocator.allocate(returns)

    assert weights.sum() == pytest.approx(1.0, abs=1e-9)
    assert 0.0 < weights["BIL"] <= 0.80
    assert allocator.last_diagnostics is not None
    assert allocator.last_diagnostics.volatility_target_applied
    assert allocator.last_diagnostics.risky_exposure == pytest.approx(
        1.0 - weights["BIL"]
    )
    assert (
        allocator.last_diagnostics.predicted_annual_volatility_after_scaling
        <= allocator.last_diagnostics.predicted_annual_volatility_before_scaling
    )


def test_volatility_target_is_not_faked_when_cash_column_is_missing() -> None:
    returns = _sample()
    allocator = HRPAlphaV1Allocator(
        HRPAlphaV1Config(
            max_weight=0.20,
            annual_target_volatility=0.05,
            cash_asset="NOT_IN_DATA",
        )
    )

    weights = allocator.allocate(returns)

    assert weights.sum() == pytest.approx(1.0, abs=1e-9)
    assert allocator.last_diagnostics is not None
    assert not allocator.last_diagnostics.volatility_target_applied
    assert allocator.last_diagnostics.risky_exposure == pytest.approx(1.0)


def test_reset_state_clears_path_dependent_data() -> None:
    allocator = HRPAlphaV1Allocator(HRPAlphaV1Config(max_weight=0.20))
    allocator.allocate(_sample())

    allocator.reset_state()

    assert allocator.last_diagnostics is None
    assert allocator.last_covariance is None
    assert allocator.last_momentum_score is None
    assert allocator._previous_weights is None


def test_shared_metrics_include_downside_and_growth_statistics() -> None:
    returns = pd.Series(
        [0.02, -0.01, 0.03, -0.04, 0.01, 0.02],
        index=pd.date_range("2025-01-03", periods=6, freq="W-FRI"),
    )
    turnover = pd.Series(0.05, index=returns.index)

    metrics = _compute_metrics(returns, turnover)

    expected = {
        "cagr",
        "sortino",
        "calmar",
        "weekly_var_95",
        "weekly_cvar_95",
        "positive_week_fraction",
    }
    assert expected.issubset(metrics)
    assert metrics["weekly_cvar_95"] <= metrics["weekly_var_95"]
    assert metrics["positive_week_fraction"] == pytest.approx(4.0 / 6.0)
