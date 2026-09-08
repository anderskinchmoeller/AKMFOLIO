import numpy as np
import pandas as pd
import pytest

from akm_hrp.allocators.ra_hrp_allocator import (
    RAHRPAllocator,
    RAHRPConfig,
    _estimate_expected_excess_returns,
)
from akm_hrp.hrp.return_adjusted import return_adjusted_hrp_allocate

_TWO_ASSET_TREE = np.array([[0.0, 1.0, 0.0, 2.0]])


def test_ra_hrp_uses_cluster_sharpe_scores_for_split():
    assets = ["low_vol", "high_return"]
    covariance = pd.DataFrame(np.diag([1.0, 4.0]), index=assets, columns=assets)
    expected_returns = pd.Series([0.10, 0.40], index=assets)

    weights, diagnostics = return_adjusted_hrp_allocate(
        expected_returns,
        covariance,
        _TWO_ASSET_TREE,
        assets,
    )

    # Cluster Sharpes are 0.10 and 0.20, hence the exact RA split is 1/3, 2/3.
    assert np.allclose(weights.to_numpy(), [1.0 / 3.0, 2.0 / 3.0])
    assert diagnostics.node_count == 1
    assert diagnostics.floor_activation_count == 0


def test_interpolation_zero_recovers_hrp_split():
    assets = ["low_vol", "high_vol"]
    covariance = pd.DataFrame(np.diag([1.0, 4.0]), index=assets, columns=assets)
    expected_returns = pd.Series([0.10, 0.10], index=assets)

    weights, _ = return_adjusted_hrp_allocate(
        expected_returns,
        covariance,
        _TWO_ASSET_TREE,
        assets,
        interpolation=0.0,
    )

    assert np.allclose(weights.to_numpy(), [0.8, 0.2])


def test_interpolation_outside_unit_interval_is_rejected():
    assets = ["A", "B"]
    covariance = pd.DataFrame(np.eye(2), index=assets, columns=assets)
    expected_returns = pd.Series([0.10, 0.20], index=assets)

    with pytest.raises(ValueError, match="between zero and one"):
        return_adjusted_hrp_allocate(
            expected_returns,
            covariance,
            _TWO_ASSET_TREE,
            assets,
            interpolation=1.1,
        )


def test_score_floor_keeps_negative_premium_split_feasible():
    assets = ["A", "B"]
    covariance = pd.DataFrame(np.eye(2), index=assets, columns=assets)
    expected_returns = pd.Series([-0.01, -0.02], index=assets)

    weights, diagnostics = return_adjusted_hrp_allocate(
        expected_returns,
        covariance,
        _TWO_ASSET_TREE,
        assets,
    )

    assert np.allclose(weights.to_numpy(), [0.5, 0.5])
    assert diagnostics.floor_activation_count == 2


def test_deep_unbalanced_tree_does_not_recurse() -> None:
    count = 1_100
    assets = [f"A{i}" for i in range(count)]
    rows = [[0.0, 1.0, 0.1, 2.0]]
    for leaf in range(2, count):
        rows.append([count + leaf - 2, leaf, 0.1 + leaf / count, leaf + 1])
    tree = np.asarray(rows, dtype=float)
    covariance = pd.DataFrame(np.eye(count), index=assets, columns=assets)
    expected_returns = pd.Series(0.01, index=assets)
    weights, diagnostics = return_adjusted_hrp_allocate(
        expected_returns, covariance, tree, assets
    )

    assert weights.sum() == pytest.approx(1.0)
    assert diagnostics.node_count == count - 1


def test_ra_hrp_follows_unbalanced_dendrogram_children():
    assets = ["A", "B", "C", "D"]
    # SciPy linkage representation of (((A, B), C), D).
    tree = np.array(
        [
            [0.0, 1.0, 0.1, 2.0],
            [4.0, 2.0, 0.2, 3.0],
            [5.0, 3.0, 0.3, 4.0],
        ]
    )
    covariance = pd.DataFrame(np.eye(4), index=assets, columns=assets)
    expected_returns = pd.Series([0.10, 0.20, 0.30, 0.40], index=assets)

    weights, diagnostics = return_adjusted_hrp_allocate(
        expected_returns,
        covariance,
        tree,
        assets,
    )

    left_score = 0.20 / np.sqrt(1.0 / 3.0)
    expected_root_left = left_score / (left_score + 0.40)
    assert np.isclose(weights[["A", "B", "C"]].sum(), expected_root_left)
    assert diagnostics.node_count == 3


def test_expected_returns_are_excess_of_weekly_risk_free_rate():
    returns = pd.DataFrame({"A": [0.01, 0.02], "B": [0.03, 0.04]})

    expected = _estimate_expected_excess_returns(
        returns,
        mean_halflife=None,
        annual_risk_free_rate=0.052,
    )

    assert np.allclose(expected.to_numpy(), [0.014, 0.034])


def test_ra_hrp_allocator_respects_project_bounds():
    rng = np.random.default_rng(41)
    common = rng.normal(0.001, 0.01, 120)
    returns = pd.DataFrame(
        np.column_stack(
            [common + rng.normal(0.0, 0.01, len(common)) for _ in range(12)]
        ),
        index=pd.date_range("2020-01-03", periods=len(common), freq="W-FRI"),
        columns=[f"asset_{i}" for i in range(12)],
    )
    allocator = RAHRPAllocator(RAHRPConfig(max_weight=0.15))

    weights = allocator.allocate(returns)

    assert np.isclose(weights.sum(), 1.0)
    assert (weights >= 0.0).all()
    assert weights.max() <= 0.15 + 1e-9
    assert allocator.last_ra_diagnostics is not None
