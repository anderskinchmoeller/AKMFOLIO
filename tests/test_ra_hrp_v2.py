import numpy as np
import pandas as pd

from akm_hrp.allocators.ra_hrp_v2_allocator import (
    RAHRPV2Allocator,
    RAHRPV2Config,
    robust_expected_excess_returns,
)
from akm_hrp.hrp.robust_return_adjusted import (
    robust_return_adjusted_hrp_allocate,
)


def _returns(seed: int = 31, assets: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    market = rng.normal(0.001, 0.018, 120)
    values = np.column_stack(
        [0.45 * market + rng.normal(i * 0.00003, 0.015, 120) for i in range(assets)]
    )
    return pd.DataFrame(
        values,
        index=pd.date_range("2020-01-03", periods=120, freq="W-FRI"),
        columns=[f"A{i:02d}" for i in range(assets)],
    )


def test_robust_expected_returns_are_finite_and_cross_sectionally_shrunk():
    returns = _returns(assets=8)
    raw = returns.ewm(halflife=26.0, adjust=False).mean().iloc[-1]

    robust = robust_expected_excess_returns(
        returns,
        mean_halflife=26.0,
        shrinkage=0.50,
        winsor_quantile=0.10,
    )

    assert np.isfinite(robust.to_numpy()).all()
    assert robust.max() - robust.min() < raw.max() - raw.min()


def test_ra_hrp_v2_is_bounded_deterministic_and_diagnostic():
    returns = _returns()
    config = RAHRPV2Config(max_weight=0.15)
    first = RAHRPV2Allocator(config)
    second = RAHRPV2Allocator(config)

    first_weights = first.allocate(returns)
    second_weights = second.allocate(returns)

    assert np.allclose(first_weights, second_weights)
    assert np.isclose(first_weights.sum(), 1.0)
    assert (first_weights >= 0.0).all()
    assert first_weights.max() <= 0.15 + 1e-9
    assert first.last_diagnostics is not None
    assert first.last_diagnostics.tree_count == 15
    assert first.last_diagnostics.allocation.scenario_count == 5
    assert first.last_diagnostics.allocation.node_count == len(first_weights) - 1


def test_ra_hrp_v2_state_reset_clears_path_dependent_diagnostics():
    allocator = RAHRPV2Allocator(RAHRPV2Config(max_weight=0.15))
    allocator.allocate(_returns())
    assert allocator.last_diagnostics is not None

    allocator.reset_state()

    assert allocator.last_diagnostics is None
    assert allocator._previous_coclustering is None
    assert allocator._previous_weights is None


def test_robust_return_adjusted_split_rewards_return_without_ignoring_risk():
    assets = ["HIGH", "LOW"]
    expected = pd.Series([0.02, 0.005], index=assets)
    covariance = pd.DataFrame(
        [[0.04, 0.0], [0.0, 0.01]],
        index=assets,
        columns=assets,
    )
    tree = np.array([[0.0, 1.0, 1.0, 2.0]])

    risk_only, _ = robust_return_adjusted_hrp_allocate(
        expected,
        covariance,
        {"base": covariance, "stress": covariance * 1.5},
        tree,
        assets,
        interpolation=0.0,
        risk_cap=1.0,
        minimum_split=0.10,
        maximum_split=0.90,
    )
    weights, diagnostics = robust_return_adjusted_hrp_allocate(
        expected,
        covariance,
        {"base": covariance, "stress": covariance * 1.5},
        tree,
        assets,
        interpolation=1.0,
        risk_cap=1.0,
        minimum_split=0.10,
        maximum_split=0.90,
    )

    assert weights["HIGH"] > risk_only["HIGH"]
    assert weights["HIGH"] < 0.50
    assert diagnostics.scenario_count == 2
    assert diagnostics.node_count == 1
