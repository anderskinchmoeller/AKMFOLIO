import numpy as np
import pandas as pd
import pytest
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from akm_hrp.allocators.schur_hrp_allocator import SchurHRPAllocator, SchurHRPConfig
from akm_hrp.hrp.schur import schur_hrp_allocate
from akm_hrp.hrp.trees import correlation_distance

_TWO_ASSET_TREE = np.array([[0.0, 1.0, 0.0, 2.0]])


def _random_covariance(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    loadings = rng.normal(size=(n, 3))
    factors = rng.normal(size=(400, 3))
    panel = factors @ loadings.T + rng.normal(size=(400, n)) * 0.8
    assets = [f"A{i}" for i in range(n)]
    return pd.DataFrame(np.cov(panel.T), index=assets, columns=assets)


def _single_linkage_tree(covariance: pd.DataFrame) -> np.ndarray:
    sd = np.sqrt(np.diag(covariance.to_numpy()))
    corr = pd.DataFrame(
        covariance.to_numpy() / np.outer(sd, sd),
        index=covariance.index,
        columns=covariance.columns,
    )
    return linkage(squareform(correlation_distance(corr), checks=False), method="single")


def _minimum_variance(covariance: pd.DataFrame) -> np.ndarray:
    raw = np.linalg.solve(covariance.to_numpy(), np.ones(len(covariance)))
    return raw / raw.sum()


def test_gamma_zero_reproduces_the_hrp_inverse_variance_split():
    assets = ["low_vol", "high_vol"]
    covariance = pd.DataFrame(np.diag([1.0, 4.0]), index=assets, columns=assets)

    weights, diagnostics = schur_hrp_allocate(
        covariance, _TWO_ASSET_TREE, assets, gamma=0.0
    )

    assert np.allclose(weights.to_numpy(), [0.8, 0.2])
    assert diagnostics.node_count == 1
    assert diagnostics.gamma_reduced_count == 0


def test_gamma_one_equals_the_unconstrained_minimum_variance_portfolio():
    # The gamma = 1 limit is the model's central claim, so it is checked on
    # several correlated universes rather than one.
    for n, seed in [(7, 0), (16, 1), (33, 2)]:
        covariance = _random_covariance(n, seed)
        tree = _single_linkage_tree(covariance)
        weights, _ = schur_hrp_allocate(
            covariance, tree, list(covariance.columns), gamma=1.0, long_only=False
        )
        np.testing.assert_allclose(
            weights.reindex(covariance.index).to_numpy(),
            _minimum_variance(covariance),
            atol=1e-9,
        )


def test_long_only_guard_keeps_every_weight_positive():
    covariance = _random_covariance(40, 4)
    tree = _single_linkage_tree(covariance)
    for gamma in (0.25, 0.5, 0.75, 1.0):
        weights, _ = schur_hrp_allocate(
            covariance, tree, list(covariance.columns), gamma=gamma
        )
        assert (weights > 0).all()
        assert weights.sum() == pytest.approx(1.0)


def test_variance_falls_as_gamma_rises():
    covariance = _random_covariance(30, 5)
    tree = _single_linkage_tree(covariance)
    variances = []
    for gamma in (0.0, 0.5, 1.0):
        weights, _ = schur_hrp_allocate(
            covariance, tree, list(covariance.columns), gamma=gamma, long_only=False
        )
        w = weights.to_numpy()
        variances.append(float(w @ covariance.to_numpy() @ w))
    assert variances[2] <= variances[1] + 1e-12
    assert variances[1] <= variances[0] + 1e-12


def test_gamma_outside_unit_interval_is_rejected():
    assets = ["A", "B"]
    covariance = pd.DataFrame(np.eye(2), index=assets, columns=assets)
    with pytest.raises(ValueError, match="between zero and one"):
        schur_hrp_allocate(covariance, _TWO_ASSET_TREE, assets, gamma=1.1)


def test_deep_unbalanced_tree_does_not_recurse() -> None:
    count = 1_100
    assets = [f"A{i}" for i in range(count)]
    rows = [[0.0, 1.0, 0.1, 2.0]]
    for leaf in range(2, count):
        rows.append([count + leaf - 2, leaf, 0.1 + leaf / count, leaf + 1])
    tree = np.asarray(rows, dtype=float)
    covariance = pd.DataFrame(np.eye(count), index=assets, columns=assets)

    for rule in ("midpoint", "dendrogram"):
        weights, diagnostics = schur_hrp_allocate(
            covariance, tree, assets, gamma=0.5, split=rule
        )
        assert weights.sum() == pytest.approx(1.0)
        assert diagnostics.node_count == count - 1


def test_default_split_halves_the_quasi_diagonal_order():
    # (((A, B), C), D) chains under single linkage. The midpoint rule ignores
    # that shape and splits the ordered leaves (A, B, C, D) into (A, B), (C, D).
    assets = ["A", "B", "C", "D"]
    tree = np.array(
        [
            [0.0, 1.0, 0.1, 2.0],
            [4.0, 2.0, 0.2, 3.0],
            [5.0, 3.0, 0.3, 4.0],
        ]
    )
    covariance = pd.DataFrame(np.diag([1.0, 1.0, 1.0, 4.0]), index=assets, columns=assets)

    weights, diagnostics = schur_hrp_allocate(covariance, tree, assets, gamma=0.0)

    # Uncorrelated: left pair has recursive variance 1/2, right pair 4/5 * 1/... ,
    # so compare against the explicit HRP split of the same ordered leaves.
    left = weights[["A", "B"]].sum()
    assert np.isclose(weights["A"], weights["B"])
    assert left > 0.5                      # the low-variance pair gets more
    assert diagnostics.split == "midpoint"
    assert diagnostics.node_count == 3


def test_dendrogram_split_concentrates_more_than_the_midpoint_split():
    # Documented trade-off: under single linkage the dendrogram rule chains and
    # concentrates. This guards the default rather than the exact numbers.
    covariance = _random_covariance(60, 11)
    tree = _single_linkage_tree(covariance)
    assets = list(covariance.columns)

    mid, _ = schur_hrp_allocate(covariance, tree, assets, gamma=0.5)
    den, _ = schur_hrp_allocate(covariance, tree, assets, gamma=0.5, split="dendrogram")

    effective = lambda w: 1.0 / float((w.to_numpy() ** 2).sum())
    assert effective(mid) > effective(den)
    assert mid.max() < den.max()


def test_unknown_split_rule_is_rejected():
    assets = ["A", "B"]
    covariance = pd.DataFrame(np.eye(2), index=assets, columns=assets)
    with pytest.raises(ValueError, match="midpoint"):
        schur_hrp_allocate(covariance, _TWO_ASSET_TREE, assets, split="binary")


def test_schur_follows_unbalanced_dendrogram_children():
    assets = ["A", "B", "C", "D"]
    # SciPy linkage representation of (((A, B), C), D).
    tree = np.array(
        [
            [0.0, 1.0, 0.1, 2.0],
            [4.0, 2.0, 0.2, 3.0],
            [5.0, 3.0, 0.3, 4.0],
        ]
    )
    covariance = pd.DataFrame(np.diag([1.0, 1.0, 1.0, 4.0]), index=assets, columns=assets)

    weights, diagnostics = schur_hrp_allocate(
        covariance, tree, assets, gamma=0.0, split="dendrogram"
    )

    # Uncorrelated assets: the tilt is inert, so each split is inverse-variance
    # on the children's own recursive variances. (A,B,C) has variance 1/3
    # against D's 4, so the left branch takes 4 / (4 + 1/3).
    expected_left = 4.0 / (4.0 + 1.0 / 3.0)
    assert np.isclose(weights[["A", "B", "C"]].sum(), expected_left)
    assert diagnostics.node_count == 3


def test_mismatched_tree_shape_is_rejected():
    assets = ["A", "B", "C"]
    covariance = pd.DataFrame(np.eye(3), index=assets, columns=assets)
    with pytest.raises(ValueError, match="tree must have shape"):
        schur_hrp_allocate(covariance, _TWO_ASSET_TREE, assets)


def test_covariance_labels_must_match_asset_names():
    assets = ["A", "B"]
    covariance = pd.DataFrame(np.eye(2), index=assets, columns=assets)
    with pytest.raises(ValueError, match="covariance labels"):
        schur_hrp_allocate(covariance, _TWO_ASSET_TREE, ["A", "C"])


def test_allocator_respects_project_bounds_and_is_deterministic():
    rng = np.random.default_rng(41)
    common = rng.normal(0.001, 0.01, 120)
    returns = pd.DataFrame(
        np.column_stack(
            [common + rng.normal(0.0, 0.01, len(common)) for _ in range(12)]
        ),
        columns=[f"A{i}" for i in range(12)],
    )
    allocator = SchurHRPAllocator(SchurHRPConfig(min_weight=0.02, max_weight=0.15))

    weights = allocator.allocate(returns)
    again = allocator.allocate(returns)

    assert weights.sum() == pytest.approx(1.0)
    assert weights.min() >= 0.02 - 1e-9
    assert weights.max() <= 0.15 + 1e-9
    pd.testing.assert_series_equal(weights, again)
    assert allocator.last_schur_diagnostics is not None
    assert allocator.last_covariance_diagnostics is not None


def test_allocator_state_reset_clears_diagnostics():
    rng = np.random.default_rng(7)
    returns = pd.DataFrame(
        rng.normal(0.0, 0.01, size=(80, 6)), columns=[f"A{i}" for i in range(6)]
    )
    allocator = SchurHRPAllocator(SchurHRPConfig(max_weight=0.5))
    allocator.allocate(returns)
    allocator.reset_state()

    assert allocator.last_schur_diagnostics is None
    assert allocator.last_covariance_diagnostics is None


def test_constant_and_missing_columns_are_dropped_before_allocation():
    rng = np.random.default_rng(3)
    returns = pd.DataFrame(
        rng.normal(0.0, 0.01, size=(90, 5)), columns=["A", "B", "C", "D", "E"]
    )
    returns["D"] = 0.0
    returns["E"] = np.nan
    weights = SchurHRPAllocator(SchurHRPConfig(max_weight=0.5)).allocate(returns)

    assert list(weights.index) == ["A", "B", "C"]
    assert weights.sum() == pytest.approx(1.0)


def test_single_asset_input_is_rejected():
    returns = pd.DataFrame({"A": [0.01, -0.02, 0.03]})
    with pytest.raises(ValueError, match="at least two assets"):
        SchurHRPAllocator().allocate(returns)
