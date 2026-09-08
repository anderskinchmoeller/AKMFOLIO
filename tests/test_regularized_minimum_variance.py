import numpy as np
import pandas as pd
import pytest

from akm_hrp.allocators.regularized_minimum_variance import (
    RegularizedMinimumVarianceAllocator,
    RegularizedMinimumVarianceConfig,
    _ledoit_wolf_dual_solution,
)
from sklearn.covariance import LedoitWolf


def _correlated_returns() -> pd.DataFrame:
    rng = np.random.default_rng(31)
    common = rng.normal(0.0, 0.01, 120)
    values = np.column_stack(
        [
            common + rng.normal(0.0, scale, len(common))
            for scale in np.linspace(0.003, 0.018, 12)
        ]
    )
    return pd.DataFrame(
        values,
        index=pd.date_range("2020-01-03", periods=len(common), freq="W-FRI"),
        columns=[f"asset_{i}" for i in range(values.shape[1])],
    )


def test_regularized_minimum_variance_is_bounded_and_well_conditioned():
    allocator = RegularizedMinimumVarianceAllocator(
        RegularizedMinimumVarianceConfig(max_weight=0.15)
    )

    weights = allocator.allocate(_correlated_returns())

    assert np.isclose(weights.sum(), 1.0)
    assert (weights >= 0.0).all()
    assert weights.max() <= 0.15 + 1e-9
    assert allocator.last_diagnostics is not None
    assert (
        allocator.last_diagnostics.condition_number_after_ridge
        < allocator.last_diagnostics.condition_number_before_ridge
    )


def test_previous_weight_shrinkage_reduces_target_change():
    returns = _correlated_returns()
    stable = RegularizedMinimumVarianceAllocator(
        RegularizedMinimumVarianceConfig(
            max_weight=0.20,
            previous_weight_shrinkage=0.75,
        )
    )
    reactive = RegularizedMinimumVarianceAllocator(
        RegularizedMinimumVarianceConfig(
            max_weight=0.20,
            previous_weight_shrinkage=0.0,
        )
    )
    initial = stable.allocate(returns.iloc[:-20])
    reactive.allocate(returns.iloc[:-20])

    stable_updated = stable.allocate(returns)
    reactive_updated = reactive.allocate(returns)

    assert (stable_updated - initial).abs().sum() < (
        reactive_updated - initial
    ).abs().sum()


def test_dual_solution_matches_primal_ledoit_wolf_system():
    observations = _correlated_returns().to_numpy()
    ridge_strength = 0.10
    precision, diagonal, shrinkage, ridge, before, after = (
        _ledoit_wolf_dual_solution(observations, ridge_strength)
    )
    estimator = LedoitWolf().fit(observations)
    covariance = estimator.covariance_
    expected_ridge = ridge_strength * np.median(np.diag(covariance))
    expected_precision = np.linalg.solve(
        covariance + expected_ridge * np.eye(covariance.shape[0]),
        np.ones(covariance.shape[0]),
    )

    assert shrinkage == pytest.approx(estimator.shrinkage_, rel=1e-9, abs=1e-12)
    assert ridge == pytest.approx(expected_ridge, rel=1e-9, abs=1e-12)
    assert np.allclose(diagonal, np.diag(covariance), rtol=1e-9, atol=1e-12)
    assert np.allclose(precision, expected_precision, rtol=1e-8, atol=1e-8)
    assert before == pytest.approx(np.linalg.cond(covariance), rel=1e-7)
    assert after == pytest.approx(
        np.linalg.cond(covariance + expected_ridge * np.eye(covariance.shape[0])),
        rel=1e-7,
    )
