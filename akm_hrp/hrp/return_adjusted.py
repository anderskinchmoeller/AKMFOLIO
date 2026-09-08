from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

_EPS = 1e-12


@dataclass(frozen=True)
class ReturnAdjustedHRPDiagnostics:
    node_count: int
    floor_activation_count: int
    mean_hrp_left_split: float
    mean_ra_left_split: float
    interpolation: float


def _cluster_moments(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    assets: list[str],
) -> tuple[float, float]:
    """Return the paper's IVP-weighted cluster mean and variance."""
    cluster_covariance = covariance.loc[assets, assets].to_numpy(dtype=float)
    variance = np.clip(np.diag(cluster_covariance), _EPS, None)
    inverse_variance = 1.0 / variance
    ivp = inverse_variance / inverse_variance.sum()
    cluster_mean = float(ivp @ expected_returns.loc[assets].to_numpy(dtype=float))
    cluster_variance = max(float(ivp @ cluster_covariance @ ivp), _EPS)
    return cluster_mean, cluster_variance


def return_adjusted_hrp_allocate(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    tree: np.ndarray,
    asset_names: list[str],
    *,
    score_floor: float = 1e-4,
    interpolation: float = 1.0,
) -> tuple[pd.Series, ReturnAdjustedHRPDiagnostics]:
    """Allocate with fixed-tree Return-Adjusted HRP.

    At each node, the RA-HRP branch probability is proportional to the
    children's floored IVP-cluster Sharpe scores. ``interpolation`` implements
    the paper's unit-free homotopy between the HRP branch probability (0) and
    the RA-HRP branch probability (1).
    """
    if score_floor <= 0.0:
        raise ValueError("score_floor must be positive.")
    interpolation = float(interpolation)
    if not 0.0 <= interpolation <= 1.0:
        raise ValueError("interpolation must be between zero and one.")

    assets = list(asset_names)
    if len(assets) < 2:
        raise ValueError("RA-HRP requires at least two assets.")
    tree = np.asarray(tree, dtype=float)
    expected_shape = (len(assets) - 1, 4)
    if tree.shape != expected_shape:
        raise ValueError(
            f"tree must have shape {expected_shape}; received {tree.shape}."
        )
    if set(covariance.index) != set(assets) or set(covariance.columns) != set(assets):
        raise ValueError("covariance labels must match asset_names.")
    if not set(assets).issubset(expected_returns.index):
        raise ValueError("expected_returns is missing one or more assets.")

    covariance = covariance.loc[assets, assets].astype(float)
    expected_returns = expected_returns.reindex(assets).astype(float)
    if not np.isfinite(covariance.to_numpy()).all():
        raise ValueError("covariance contains non-finite values.")
    if not np.isfinite(expected_returns.to_numpy()).all():
        raise ValueError("expected_returns contains non-finite values.")

    weights = pd.Series(1.0, index=assets, dtype=float)
    hrp_splits: list[float] = []
    ra_splits: list[float] = []
    floor_activations = 0

    # SciPy linkage rows create cluster ids n, n+1, ... in order. Retaining
    # these child relationships is essential: a midpoint split of the ordered
    # leaves describes a different tree whenever the dendrogram is unbalanced.
    n_assets = len(assets)
    children: dict[int, tuple[int, int]] = {}
    members: dict[int, list[str]] = {
        leaf_id: [asset] for leaf_id, asset in enumerate(assets)
    }
    member_positions: dict[int, list[int]] = {
        leaf_id: [leaf_id] for leaf_id in range(n_assets)
    }
    covariance_values = covariance.to_numpy(dtype=float)
    expected_values = expected_returns.to_numpy(dtype=float)
    inverse_variance = 1.0 / np.clip(
        np.diag(covariance_values), _EPS, None
    )
    # Cache the sufficient statistics for every tree cluster. Re-slicing a
    # growing covariance submatrix at every node is cubic on a chain-like tree;
    # combining child statistics evaluates every cross-asset pair only once.
    mass: dict[int, float] = {
        leaf_id: float(inverse_variance[leaf_id])
        for leaf_id in range(n_assets)
    }
    return_numerator: dict[int, float] = {
        leaf_id: float(inverse_variance[leaf_id] * expected_values[leaf_id])
        for leaf_id in range(n_assets)
    }
    risk_numerator: dict[int, float] = {
        leaf_id: float(
            inverse_variance[leaf_id] ** 2
            * covariance_values[leaf_id, leaf_id]
        )
        for leaf_id in range(n_assets)
    }
    moments: dict[int, tuple[float, float]] = {
        leaf_id: (
            float(expected_values[leaf_id]),
            float(covariance_values[leaf_id, leaf_id]),
        )
        for leaf_id in range(n_assets)
    }
    for row_number, row in enumerate(tree):
        node_id = n_assets + row_number
        left_id, right_id = int(row[0]), int(row[1])
        if left_id not in members or right_id not in members:
            raise ValueError("tree contains an invalid or forward child reference.")
        if set(members[left_id]).intersection(members[right_id]):
            raise ValueError("tree child clusters overlap.")
        children[node_id] = (left_id, right_id)
        members[node_id] = members[left_id] + members[right_id]
        left_positions = member_positions[left_id]
        right_positions = member_positions[right_id]
        member_positions[node_id] = left_positions + right_positions
        cross = float(
            inverse_variance[left_positions]
            @ covariance_values[np.ix_(left_positions, right_positions)]
            @ inverse_variance[right_positions]
        )
        mass[node_id] = mass[left_id] + mass[right_id]
        return_numerator[node_id] = (
            return_numerator[left_id] + return_numerator[right_id]
        )
        risk_numerator[node_id] = (
            risk_numerator[left_id] + risk_numerator[right_id] + 2.0 * cross
        )
        moments[node_id] = (
            return_numerator[node_id] / mass[node_id],
            max(risk_numerator[node_id] / mass[node_id] ** 2, _EPS),
        )

    root_id = n_assets + len(tree) - 1
    if set(members[root_id]) != set(assets):
        raise ValueError("tree root does not contain every asset exactly once.")

    # A chain-like linkage can be thousands of nodes deep. Traverse it
    # iteratively so a valid dendrogram cannot hit Python's recursion limit.
    pending = [root_id]
    while pending:
        node_id = pending.pop()
        if node_id < n_assets:
            continue

        left_id, right_id = children[node_id]
        left = members[left_id]
        right = members[right_id]
        left_mean, left_variance = moments[left_id]
        right_mean, right_variance = moments[right_id]

        left_raw_score = left_mean / np.sqrt(left_variance)
        right_raw_score = right_mean / np.sqrt(right_variance)
        floor_activations += int(left_raw_score < score_floor)
        floor_activations += int(right_raw_score < score_floor)
        left_score = max(left_raw_score, score_floor)
        right_score = max(right_raw_score, score_floor)

        # HRP assigns more capital to the child with lower cluster variance.
        hrp_left = right_variance / (left_variance + right_variance)
        ra_left = left_score / (left_score + right_score)
        left_split = (1.0 - interpolation) * hrp_left + interpolation * ra_left

        weights.loc[left] *= left_split
        weights.loc[right] *= 1.0 - left_split
        hrp_splits.append(float(hrp_left))
        ra_splits.append(float(ra_left))

        pending.append(right_id)
        pending.append(left_id)
    weights = weights.clip(lower=0.0)
    weights /= weights.sum()
    diagnostics = ReturnAdjustedHRPDiagnostics(
        node_count=len(ra_splits),
        floor_activation_count=floor_activations,
        mean_hrp_left_split=float(np.mean(hrp_splits)) if hrp_splits else 0.0,
        mean_ra_left_split=float(np.mean(ra_splits)) if ra_splits else 0.0,
        interpolation=interpolation,
    )
    return weights, diagnostics
