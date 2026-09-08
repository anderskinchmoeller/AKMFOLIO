import numpy as np
import pandas as pd
import pytest

from akm_hrp.allocators.barra_factor_hrp import (
    BarraFactorHRPAllocator,
    BarraFactorHRPConfig,
    barra_factor_covariance,
)


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
