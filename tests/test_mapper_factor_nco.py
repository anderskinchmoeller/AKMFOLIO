import numpy as np
import pandas as pd
import pytest

from akm_hrp.allocators.mapper_factor_nco import (
    MapperFactorNCOAllocator,
    MapperFactorNCOConfig,
    fit_mapper_factor_model,
    hierarchical_quasi_diagonal_clusters,
    nested_clustered_optimization,
)


def _factor_sample(
    observations: int = 160,
    assets: int = 12,
    seed: int = 17,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2020-01-03", periods=observations, freq="W-FRI")
    factors = rng.normal(
        loc=[0.0010, 0.0005, 0.0002],
        scale=[0.020, 0.015, 0.010],
        size=(observations, 3),
    )
    loadings = rng.normal(size=(3, assets))
    residuals = rng.normal(scale=0.015, size=(observations, assets))
    returns = factors @ loadings + residuals
    return (
        pd.DataFrame(
            returns,
            index=index,
            columns=[f"asset_{i}" for i in range(assets)],
        ),
        pd.DataFrame(
            factors,
            index=index,
            columns=["market", "value", "momentum"],
        ),
    )


@pytest.mark.parametrize("use_external_factors", [False, True])
def test_allocator_returns_bounded_fully_invested_weights(
    use_external_factors: bool,
) -> None:
    returns, factors = _factor_sample()
    allocator = MapperFactorNCOAllocator(
        MapperFactorNCOConfig(max_weight=0.20),
        factor_returns=factors if use_external_factors else None,
    )

    weights = allocator.allocate(returns)

    assert weights.index.equals(returns.columns)
    assert np.isfinite(weights).all()
    assert weights.sum() == pytest.approx(1.0, abs=1e-9)
    assert weights.min() >= -1e-12
    assert weights.max() <= 0.20 + 1e-9
    assert allocator.last_diagnostics is not None
    assert allocator.last_diagnostics.factor_model.source == (
        "external" if use_external_factors else "statistical_pca"
    )
    assert allocator.last_diagnostics.mapper.regime_observation_count >= 26
    assert allocator.last_diagnostics.nco.optimizer_failures == 0


def test_factor_covariance_is_positive_semidefinite() -> None:
    returns, factors = _factor_sample()
    expected, covariance, factor_diagnostics, mapper_diagnostics = (
        fit_mapper_factor_model(
            returns,
            MapperFactorNCOConfig(),
            factor_returns=factors,
        )
    )

    eigenvalues = np.linalg.eigvalsh(covariance.to_numpy(dtype=float))
    assert expected.index.equals(returns.columns)
    assert np.isfinite(expected).all()
    assert eigenvalues.min() > 0.0
    assert factor_diagnostics.factor_count == 3
    assert mapper_diagnostics.node_count > 0
    assert mapper_diagnostics.effective_observation_count > 1.0


def test_external_factor_alignment_cannot_see_future_rows() -> None:
    returns, factors = _factor_sample()
    future_index = pd.date_range(
        returns.index[-1] + pd.Timedelta(weeks=1),
        periods=20,
        freq="W-FRI",
    )
    rng = np.random.default_rng(99)
    future = pd.DataFrame(
        rng.normal(size=(20, 3)),
        index=future_index,
        columns=factors.columns,
    )
    factors_with_future = pd.concat([factors, future])
    config = MapperFactorNCOConfig(max_weight=0.20)

    trimmed = MapperFactorNCOAllocator(config, factor_returns=factors).allocate(returns)
    with_future = MapperFactorNCOAllocator(
        config,
        factor_returns=factors_with_future,
    ).allocate(returns)

    pd.testing.assert_series_equal(trimmed, with_future, atol=1e-11, rtol=1e-11)


def test_hierarchy_produces_a_quasi_diagonal_permutation() -> None:
    returns, _ = _factor_sample(assets=9)
    covariance = returns.cov()
    clusters, order = hierarchical_quasi_diagonal_clusters(
        covariance,
        MapperFactorNCOConfig(max_clusters=5),
    )

    assert sorted(order) == sorted(covariance.columns)
    assert sorted(asset for cluster in clusters for asset in cluster) == sorted(
        covariance.columns
    )
    assert 2 <= len(clusters) <= 5


def test_nco_expected_return_tilt_changes_the_portfolio() -> None:
    assets = ["alpha", "b", "c", "d"]
    covariance = pd.DataFrame(np.eye(4), index=assets, columns=assets)
    expected = pd.Series([0.02, 0.0, 0.0, 0.0], index=assets)
    config = MapperFactorNCOConfig(
        max_weight=0.80,
        expected_return_strength=0.50,
        weight_anchor_strength=0.0,
        turnover_penalty=0.0,
        risk_contribution_cap=1.0,
    )

    weights, diagnostics = nested_clustered_optimization(
        expected,
        covariance,
        config,
    )

    assert weights["alpha"] > weights.drop("alpha").max()
    assert diagnostics.optimizer_failures == 0
    assert diagnostics.quasi_diagonal_order


def test_reset_state_removes_previous_target_and_diagnostics() -> None:
    returns, factors = _factor_sample()
    allocator = MapperFactorNCOAllocator(
        MapperFactorNCOConfig(max_weight=0.20),
        factor_returns=factors,
    )
    allocator.allocate(returns)

    allocator.reset_state()

    assert allocator.last_diagnostics is None
    assert allocator.last_expected_returns is None
    assert allocator.last_factor_covariance is None
    assert allocator._previous_weights is None
